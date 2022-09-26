from typing import Any, Dict, Union

import numpy as np
import torch.nn.functional
import torch.utils.checkpoint as checkpoint
from cbmi_utils.pytorch.tools.visualizations import plot_UMAP, plot_TSNE
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from apex.optimizers import FusedLAMB
from determined.pytorch import (DataLoader, LRScheduler, PyTorchTrial, PyTorchTrialContext, TorchData)

from data.augmentation_wrapper import AugmentationWrapper
from models.util_layers import projection_head
from models.resnet import SimClrResNet
from models.wide_resnet import SimClrWideResNet
from utils import datasets
from utils.global_parameters import get_global_parameters
from utils.loss import ContrastiveLoss
from utils.transformation_loader import load_transforms


NORMALIZE_LOSS = True
NUM_MODEL_CHECKPOINTS = 0  # try to optimize model memory usage. See: https://pytorch.org/docs/stable/checkpoint.html


class ContrastiveTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        super().__init__(context)
        self.context = context
        self.global_parameters = get_global_parameters()['contrastive']

        self.tblogger = TorchWriter()  # Tensorboard log

        self.dataset_name = self.context.get_hparam('dataset')

        # Creates a feature vector
        head_width = self.context.get_hparam('head_width')
        embedding_size = self.context.get_hparam('embedding_size')
        encoder_model = self.context.get_hparam('model')
        if 'wide' in encoder_model:
            encoder = SimClrWideResNet(
                feature_dim=embedding_size,
                encoder=encoder_model,
            )
        else:
            encoder = SimClrResNet(
                feature_dim=embedding_size,
                encoder=encoder_model,
            )

        # pred layer
        encoder.pred = projection_head(input_shape=embedding_size, num_outputs=128, head_width=head_width)

        self.encoder = self.context.wrap_model(encoder)

        optimizer_name = self.context.get_hparam('optimizer')
        if optimizer_name == 'lamb':
            optimizer = FusedLAMB(
                self.encoder.parameters(),
                lr=self.context.get_hparam('learning_rate'),
                weight_decay=self.context.get_hparam('l2_regularization')
            )
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.encoder.parameters(),
                lr=self.context.get_hparam('learning_rate'),
                momentum=0.9,
                weight_decay=self.context.get_hparam('l2_regularization')
            )
        else:
            raise ValueError('Could not find optimizer "{}"'.format(optimizer_name))
        self.optimizer = self.context.wrap_optimizer(optimizer)

        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.context.get_experiment_config()['searcher']['max_length']['batches']
        )
        self.scheduler = self.context.wrap_lr_scheduler(scheduler, LRScheduler.StepMode.STEP_EVERY_BATCH)

        self.train_loss_func = ContrastiveLoss(tau=self.context.get_hparam('tau'), normalize=NORMALIZE_LOSS)
        self.eval_loss_func = ContrastiveLoss(tau=0.5, normalize=NORMALIZE_LOSS)

        self.current_train_batch = 0
        self.full_eval_steps = [int(s) for s in self.context.get_hparam('full_eval_steps').split(':')]

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Union[torch.Tensor, Dict[str, Any]]:
        self.current_train_batch = batch_idx

        # with torch.autograd.detect_anomaly():  # only use for debugging
        x_i = batch['augmented0']
        x_j = batch['augmented1']

        # try to optimize memory usage:
        if NUM_MODEL_CHECKPOINTS:
            z_i = checkpoint.checkpoint_sequential(self.encoder, NUM_MODEL_CHECKPOINTS, x_i)
            z_j = checkpoint.checkpoint_sequential(self.encoder, NUM_MODEL_CHECKPOINTS, x_j)
        else:
            z_i = self.encoder(x_i)
            z_j = self.encoder(x_j)

        loss = self.train_loss_func(z_i, z_j)

        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        return {
            'loss': loss,
        }

    def evaluate_full_dataset(self, data_loader: DataLoader) -> Dict[str, Any]:
        # noinspection PyProtectedMember
        batch_idx = self.context._current_batch_idx + 1

        with torch.no_grad():
            outputs = []
            labels = []
            losses = []

            for batch in data_loader:
                # TODO: we evaluate on augmented images?
                x_i, x_j, y = batch['augmented0'], batch['augmented1'], batch['label']
                x_i, x_j, y = self.context.to_device(x_i), self.context.to_device(x_j), self.context.to_device(y)

                z_i, z_j = self.encoder(x_i), self.encoder(x_j)

                loss = self.eval_loss_func(z_i, z_j)
                loss = loss.cpu().numpy().tolist()

                outputs.append(z_i)
                labels.append(y)
                losses.append(loss)

            outputs = torch.cat(outputs)
            labels = torch.cat(labels)

        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        if self.full_eval_steps and self.current_train_batch >= self.full_eval_steps[0]:
            del self.full_eval_steps[0]
            self.tblogger.writer.add_figure(
                f'UMAP_batch_{batch_idx}',
                plot_UMAP(outputs, labels, title=f'UMAP (batch={batch_idx})'),
                global_step=batch_idx
            )
            self.tblogger.writer.add_figure(
                f'TNSE_batch_{batch_idx}',
                plot_TSNE(outputs, labels, title=f'tSNE (batch={batch_idx})'),
                global_step=batch_idx
            )

            if self.context.get_hparam('use_knn'):
                knn_classifier = self._create_knn_classifier()
                knn_preds = knn_classifier.predict(outputs)
                from sklearn.metrics import accuracy_score
                knn_accuracy = accuracy_score(knn_preds, labels)

                return {"v_loss": np.mean(losses), "knn_accuracy": knn_accuracy}

        return {
            'v_loss': np.mean(losses)
        }

    '''
    def evaluate_batch(self, batch: TorchData, **kwargs) -> Dict[str, Any]:
        x_i = batch['augmented0']
        x_j = batch['augmented1']

        z_i = self.encoder(x_i)
        z_j = self.encoder(x_j)

        loss = self.eval_loss_func(z_i, z_j)

        return {
            'loss': loss,
        }
    '''

    def build_training_data_loader(self) -> DataLoader:
        dataset = datasets.get_dataset(self.dataset_name, self.global_parameters, 'train')

        augmentation_wrapper = AugmentationWrapper(dataset, self._get_transforms())

        dataloader = DataLoader(
            augmentation_wrapper,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=self.context.get_hparam('workers'),
            pin_memory=True
        )

        return dataloader

    def build_validation_data_loader(self) -> DataLoader:
        dataset = datasets.get_dataset(self.dataset_name, self.global_parameters, 'valid')

        augmentation_wrapper = AugmentationWrapper(dataset, self._get_transforms())

        dataloader = DataLoader(
            augmentation_wrapper,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=self.context.get_hparam('workers'),
            pin_memory=True
        )

        return dataloader

    def _get_slide_size(self):
        image_shape = self.global_parameters['image_shape']
        return image_shape['width'], image_shape['height']

    def _get_transforms(self):
        image_rescale_size = self.context.get_hparam('image_rescale_size')
        return load_transforms(
            (
                self.context.get_hparam('augmentation1'),
                self.context.get_hparam('augmentation2'),
            ),
            self._get_slide_size(),
            use_rotation=True,
            image_rescale_size=(image_rescale_size, image_rescale_size),  # Resize at the end of augmentation stack
        )

    def _create_knn_classifier(self):
        # Get the train data again without TwoCrops
        dataset = datasets.get_dataset(self.dataset_name, self.global_parameters, 'train')
        one_transform = [self._get_transforms()[0]]
        augmentation_wrapper = AugmentationWrapper(dataset, one_transform)

        dataloader = DataLoader(
            augmentation_wrapper,
            batch_size=2 * self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=self.context.get_hparam('workers'),
            pin_memory=True,
            drop_last=True,
        )

        # Forward the whole set
        with torch.no_grad():
            x_hat = []
            y = []

            for batch in dataloader:
                x, label = batch['augmented0'], batch['label']
                x = self.context.to_device(x)

                output = self.encoder(x)

                x_hat.append(output.detach().cpu())
                y.append(label.detach().cpu())

        x_hat = torch.cat(x_hat)
        y = torch.cat(y)
        x_hat = x_hat.numpy()
        y = y.numpy()

        # Train a kNN on the embeddings
        from sklearn.neighbors import KNeighborsClassifier
        knn_classifier = KNeighborsClassifier(n_neighbors=5)
        knn_classifier.fit(x_hat, y)

        return knn_classifier
