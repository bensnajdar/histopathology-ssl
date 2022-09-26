import os
from typing import Union, Dict, Any

import numpy as np
import torch.utils.data
import torchvision
import umap
import umap.plot
import seaborn as sns
from sklearn.manifold import TSNE
from determined import pytorch
from determined.pytorch import PyTorchTrial, DataLoader
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from matplotlib import pyplot as plt

from data.augmentation_wrapper import AugmentationWrapper
from models.resnet import SimClrResNet
from models.wide_resnet import SimClrWideResNet
from utils import datasets
from utils.global_parameters import get_global_parameters


class VisualizeEmbeddingsTrial(PyTorchTrial):
    def __init__(self, context: pytorch.PyTorchTrialContext) -> None:
        super().__init__(context)
        self.context = context

        self.tblogger = TorchWriter()  # Tensorboard log

        self.dataset_name = context.get_hparam('dataset')
        self.global_parameters = get_global_parameters()['contrastive']
        self.encoder_model = self.context.get_hparam('model')
        self.plot_index = 0

        # We load the encoder by accessing the checkpoints directly on disc
        checkpoint_uuid = self.context.get_hparam('checkpoint_uuid')
        encoder_path = os.path.join('/checkpoints', checkpoint_uuid, 'state_dict.pth')
        if self.encoder_model == 'resnet':
            encoder = resnet18_from_state_dict(encoder_path, use_softmax=False, use_sigmoid=False)  # TODO
        elif self.encoder_model == 'wide_resnet':
            encoder = wide_resnet_with_dense_from_state_dict(encoder_path)
        elif self.encoder_model == 'alex_net':
            encoder = alex_net_from_state_dict(encoder_path, use_softmax=False, use_sigmoid=False)
        else:
            raise ValueError('encoder_model has invalid value: {}'.format(self.encoder_model))
        self.encoder = self.context.wrap_model(encoder)

        # self.optimizer is not used, but determined demands one
        self.optimizer = self.context.wrap_optimizer(torch.optim.SGD(self.encoder.parameters(), lr=0.0))

    def train_batch(self, batch: pytorch.TorchData, epoch_idx: int, batch_idx) -> Union[torch.Tensor, Dict[str, Any]]:
        return {'accuracy': 0.0}

    def evaluate_full_dataset(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        # create embeddings
        simclr_embeddings = []
        labels = []

        with torch.no_grad():
            for batch in data_loader:
                image = self.context.to_device(batch['augmented0'])  # TODO: why do I have to call to_device here?
                if isinstance(self.encoder, WideResnetWithDense):
                    output = self.encoder(image, ignore_first_head_layer=True)
                else:
                    output = self.encoder(image)
                simclr_embeddings.append(output.cpu().numpy())
                labels.append(batch['label'].cpu().numpy())

        simclr_embeddings = np.concatenate(simclr_embeddings)
        labels = np.concatenate(labels)

        # umap
        reducer = umap.UMAP()
        umap_projection = reducer.fit(simclr_embeddings)

        umap_plot = umap.plot.points(umap_projection, labels=labels, theme='fire')
        self.tblogger.writer.add_figure(
            f'encoder_space_umap_{self.plot_index}', umap_plot.figure, global_step=self.plot_index
        )

        # tsne
        tsne_projection = TSNE(n_components=2).fit_transform(simclr_embeddings)

        fig = plt.figure(figsize=(10, 10))
        plt.scatter(
            tsne_projection[:, 0],
            tsne_projection[:, 1],
            s=3,
            c=[sns.color_palette()[x] for x in labels]
        )
        plt.gca().set_aspect('equal', 'datalim')
        plt.grid()
        plt.title('TSNE projection of SimCLR for {} dataset'.format(self.dataset_name), fontsize=18)
        self.tblogger.writer.add_figure(
            f'encoder_space_tsne_{self.plot_index}', fig, global_step=self.plot_index
        )

        self.plot_index += 1

        return {'accuracy': 0.0}

    def build_training_data_loader(self) -> pytorch.DataLoader:
        dataset = datasets.get_dataset(self.dataset_name, self.global_parameters, 'train')

        augmentation_wrapper = AugmentationWrapper(dataset, transforms=[torchvision.transforms.ToTensor()])

        dataloader = DataLoader(
            augmentation_wrapper,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=self.context.get_hparam('workers'),
            pin_memory=True
        )

        return dataloader

    def build_validation_data_loader(self) -> pytorch.DataLoader:
        dataset = datasets.get_dataset(self.dataset_name, self.global_parameters, 'valid')

        augmentation_wrapper = AugmentationWrapper(dataset, transforms=[torchvision.transforms.ToTensor()])

        dataloader = DataLoader(
            augmentation_wrapper,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=False,
            num_workers=self.context.get_hparam('workers'),
            pin_memory=True
        )

        return dataloader
