import logging
from typing import Dict, Any

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader, TorchData, LRScheduler
from determined.tensorboard.metric_writers.pytorch import TorchWriter

from src.simsiam import SimSiamLoss, TwoCropsTransform
from src.simtriplet import SimTripletLoss
from src.init import init_data, init_model
import src.augmentations as augs
from cbmi_utils.pytorch.optim.larc import LARC
from cbmi_utils.pytorch.tools.visualizations import plot_UMAP, plot_TSNE
from cbmi_utils.pytorch.tools.logging import print_learnable_params


class SimSiam(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.logger = logging.getLogger()  # Console log
        self.tblogger = TorchWriter()  # Tensorboard log

        # -- Model
        model = init_model(
            self.context.get_hparam('method'),
            self.context.get_hparam('encoder'),
            self.context.get_hparam('feature_dim'),
            self.context.get_hparam('pred_hidden_dim')
        )

        self.logger.info(model)
        self.model = self.context.wrap_model(model)

        # -- Optimizer
        lr = self.context.get_hparam('lr') * self.context.get_global_batch_size() / 256
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr,
            momentum=self.context.get_hparam('momentum'),
            weight_decay=self.context.get_hparam('weight_decay')
        )
        if self.context.get_hparam('use_lars'):
            optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)
        self.optimizer = self.context.wrap_optimizer(optimizer)

        # -- Scheduler
        if self.context.get_hparam('use_scheduler'):
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.context.get_experiment_config()['searcher']['max_length']['batches']
            )
            self.scheduler = self.context.wrap_lr_scheduler(scheduler, LRScheduler.StepMode.STEP_EVERY_BATCH)

        # -- Loss function
        method = self.context.get_hparam('method')
        if method == 'simsiam':
            self.loss = SimSiamLoss()
            self.num_samples = 1
        elif method == 'simtriplet':
            self.loss = SimTripletLoss()
            self.num_samples = 2
        else:
            raise NotImplementedError(f'No such method {method} implemented!')

    def build_training_data_loader(self) -> DataLoader:
        augmentations = augs.__dict__[self.context.get_hparam('train_augmentation')](
            self.context.get_hparam('dataset'),
            self.context.get_hparam('normalize_data'),
            self.context.get_hparam('img_rescale_size')
        )

        dataset = init_data(self.context.get_hparam('dataset'), sub_set='train', transform=TwoCropsTransform(augmentations), num_samples=self.num_samples)

        dataloader = DataLoader(dataset,
                                batch_size=self.context.get_per_slot_batch_size(),
                                num_workers=self.context.get_data_config()['worker'],
                                pin_memory=True,
                                shuffle=True,
                                drop_last=True)

        return dataloader

    def build_validation_data_loader(self) -> DataLoader:
        augmentations = augs.__dict__[self.context.get_hparam('val_augmentation')](
            self.context.get_hparam('dataset'),
            self.context.get_hparam('normalize_data'),
            self.context.get_hparam('img_rescale_size')
        )

        dataset = init_data(self.context.get_hparam('dataset_val'), sub_set='valid', transform=augmentations)

        dataloader = DataLoader(dataset,
                                batch_size=self.context.get_per_slot_batch_size(),
                                num_workers=self.context.get_data_config()['worker'],
                                pin_memory=True,
                                shuffle=False)

        return dataloader

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        if self.context.get_hparam('freeze_pred') == batch_idx:
            for name, param in self.model.named_parameters():
                if name in ['pred.bn1.weight', 'pred.bn1.bias', 'pred.fc1.weight', 'pred.fc2.weight', 'pred.fc2.bias']:
                    param.requires_grad = False
                    print(f'Freezed predictor from now on: {print_learnable_params(self.model)}')

        if self.context.get_hparam('method') == 'simsiam':
            x1, x2 = batch[0][0], batch[0][1]
            p1, p2, z1, z2 = self.model(x1, x2)
            t_loss = self.loss(p1, p2, z1, z2)
        elif self.context.get_hparam('method') == 'simtriplet':
            x1, x2, x3 = batch[0][0], batch[0][1], batch[1][0]
            p1, p2, p3, z1, z2, z3 = self.model(x1, x2, x3)
            t_loss = self.loss(p1, p2, p3, z1, z2, z3)
        else:
            raise NotImplementedError('FuManSchuh')

        # Define the training backward pass and step the optimizer.
        self.context.backward(t_loss)
        self.context.step_optimizer(self.optimizer)

        # -- log train loss also for the val step without graph stuff
        self.t_loss = t_loss.item()

        z1_normed = torch.nn.functional.normalize(z1)
        stats = {"t_loss": self.t_loss, 'std_z_normed': torch.std(z1_normed)}

        if self.context.get_hparam('use_scheduler'):
            stats['lr'] = self.scheduler.get_last_lr()[0]
        return stats

    def evaluate_full_dataset(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        batch_idx = self.context._current_batch_idx + 1

        if self.context.get_hparam('plot_specific_iterations') is not None:
            condition = batch_idx % self.context.get_hparam('additional_eval') == 0 or str(batch_idx) in self.context.get_hparam('plot_specific_iterations').split()
        else:
            condition = batch_idx % self.context.get_hparam('additional_eval') == 0

        if condition:
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    self.tblogger.writer.add_histogram(name, param, batch_idx)

            with torch.no_grad():
                outputs = []
                labels = []

                for batch in data_loader:
                    x, label = batch[0], batch[1]
                    x = self.context.to_device(x)

                    x_hat = self.model(x)

                    outputs.append(x_hat.detach().cpu())
                    labels.append(label.detach().cpu())

                outputs = torch.cat(outputs)
                labels = torch.cat(labels)
            outputs = outputs.numpy()
            labels = labels.numpy()

            self.tblogger.writer.add_figure(
                f'encoder_space_UMAP_{batch_idx}',
                plot_UMAP(feature_vec=outputs, label_vec=labels),
                global_step=batch_idx
            )

            self.tblogger.writer.add_figure(
                f'encoder_space_TNSE_batch_{batch_idx}',
                plot_TSNE(feature_vec=outputs, label_vec=labels),
                global_step=batch_idx
            )

            # Train a kNN and predict the val mappings
            if self.context.get_hparam('dataset') != 'imagenet' and self.context.get_hparam('use_knn') is True:
                knn_classifier = self._create_knn_classifier()
                knn_preds = knn_classifier.predict(outputs)
                from sklearn.metrics import accuracy_score
                knn_acc = accuracy_score(knn_preds, labels)

                return {"v_loss": self.t_loss, "knn_acc": knn_acc}

        return {"v_loss": self.t_loss}

    def _create_knn_classifier(self):
        # Get the train data again without TwoCrops
        augmentations = augs.__dict__[self.context.get_hparam('train_augmentation')](
            self.context.get_hparam('dataset'),
            self.context.get_hparam('normalize_data'),
            self.context.get_hparam('img_rescale_size')
        )

        t_ds = init_data(self.context.get_hparam('dataset'), sub_set='train', transform=augmentations)

        dataloader = DataLoader(
            t_ds,
            batch_size=2 * self.context.get_per_slot_batch_size(),
            num_workers=self.context.get_data_config()['worker'],
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )

        # Forward the whole set
        with torch.no_grad():
            x_hat = []
            y = []

            for batch in dataloader:
                x, label = batch[0], batch[1]
                x = self.context.to_device(x)

                output = self.model(x)

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
