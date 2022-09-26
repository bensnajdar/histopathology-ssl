import logging
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional
import torchvision.models as models
import torchmetrics.functional.classification as metrics
from torch.optim.lr_scheduler import CosineAnnealingLR

from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader, TorchData, LRScheduler
from determined.tensorboard.metric_writers.pytorch import TorchWriter

from simsiam.src.simsiam import finetune_classifier
from simsiam.src.init import init_data, init_model
import simsiam.src.augmentations as augs
import paws.src.densenet as densenet
import paws.src.wide_resnet as wide_resnet
import paws.src.resnet as resnet
from clr.src.models.wide_resnet import SimClrWideResNet
from clr.src.models.util_layers import finetune_classifier as simclr_finetune_classifier
from clr.src.models.resnet import SimClrResNet

from cbmi_utils.pytorch.tools.visualizations import create_conf_matrix_plot, plot_UMAP, plot_TSNE
from cbmi_utils.pytorch.optim.larc import LARC


# each method get its own init function
def init_paws_model(model_name='resnet50', output_dim=128, pretrained_base_model_path=None, state_dict=None,
                    use_pred=True, number_classes=2, freeze_encoder: bool = False, dropout_rate=0.0, pred_head_structure='three_layer_mlp'):

    """Initialize a model from paws training

    Args:
        model_name (str, optional): For valid inputs see `src/resnet.py` or `src/wide_resnet.py`. Defaults to 'resnet50'.
        use_pred (bool, optional): Adds a prediction head to the network. Do not use this during training! Defaults to False.
        output_dim (int, optional): Output size of the projection head. Defaults to 128.

    Returns:
        [type]: [description]
    """
    # NOTE: 'hidden_dim' aka size of the z-space can't be configured atm
    if 'densenet' in model_name:
        encoder = densenet.__dict__[model_name]()
        hidden_dim = encoder.hidden_dim
    elif 'wide_resnet' in model_name:
        encoder = wide_resnet.__dict__[model_name](dropout_rate=dropout_rate)
        hidden_dim = 128
        if 'w4' in model_name:
            hidden_dim *= 2
        elif 'w8' in model_name:
            hidden_dim *= 4
    else:
        encoder = resnet.__dict__[model_name]()
        hidden_dim = 2048
        if 'w2' in model_name:
            hidden_dim *= 2
        elif 'w4' in model_name:
            hidden_dim *= 4
        elif '18' in model_name:
            hidden_dim = 512

    # load pre_trained weigths into feature encoding part (base model)
    if pretrained_base_model_path:
        state_dict = torch.load(pretrained_base_model_path)['models_state_dict']
        encoder.load_state_dict(state_dict[0])

    # -- add paws projection head
    encoder.fc = torch.nn.Sequential(OrderedDict([
        ('fc1', torch.nn.Linear(hidden_dim, hidden_dim, bias=False)),
        ('bn1', torch.nn.BatchNorm1d(hidden_dim)),
        ('relu1', torch.nn.ReLU(inplace=True)),
        ('fc2', torch.nn.Linear(hidden_dim, hidden_dim, bias=False)),
        ('bn2', torch.nn.BatchNorm1d(hidden_dim)),
        ('relu2', torch.nn.ReLU(inplace=True)),
        ('fc3', torch.nn.Linear(hidden_dim, output_dim, bias=False))
    ]))

    # load pre_trained paws model
    if state_dict:
        encoder.load_state_dict(state_dict[0])

    # add prediction head for finetuning downstream tasks
    if use_pred:
        '''
        encoder.pred = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(output_dim, output_dim // 2)),
            ('bn1', torch.nn.BatchNorm1d(output_dim // 2)),
            ('relu1', torch.nn.ReLU(inplace=True)),
            ('fc2', torch.nn.Linear(output_dim // 2, output_dim // 4)),
            ('bn2', torch.nn.BatchNorm1d(output_dim // 4)),
            ('relu2', torch.nn.ReLU(inplace=True)),
            ('fc3', torch.nn.Linear(output_dim // 4, number_classes))
        ]))
        '''

        encoder.pred = finetune_classifier(output_dim, number_classes, output_dim, pred_head_structure)

    if freeze_encoder is not False:
        for param in encoder.parameters():
            param.requires_grad = False
        for param in encoder.pred.parameters():
            param.requires_grad = True  # uses only prediction head for finetuning

    return encoder


def init_simclr_model(
        encoder: str,
        class_num: int,
        state_dict: Optional[dict],
        freeze_encoder: bool = False,
        keep_enc_fc: bool = True,
        pred_head_structure: str = None,
        pred_head_features: int = 256,
):
    enc_fc_out_dim = 128
    if state_dict is not None:
        state_dict = state_dict[0]
        enc_fc_out_dim = state_dict['fc.0.weight'].size()[0]

    if encoder == 'wide_resnet28w2':
        model = SimClrWideResNet(feature_dim=enc_fc_out_dim)
    elif encoder in ('resnet18', 'resnet50'):
        model = SimClrResNet(
            feature_dim=enc_fc_out_dim,
            encoder=encoder,
        )
    else:
        raise ValueError('Could not create model {}'.format(encoder))

    if state_dict is not None:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        assert not missing_keys

    if freeze_encoder:
        for param in model.parameters():
            param.requires_grad = False

    if keep_enc_fc:
        # use model fc layer from checkpoint; fc layer is not trained
        model.pred = simclr_finetune_classifier(enc_fc_out_dim, class_num, pred_head_features, pred_head_structure)
    else:
        prev_dim = model.get_pre_fc_dim()
        model.fc = simclr_finetune_classifier(
            prev_dim, class_num, pred_head_features, pred_head_structure
        )
        model.pred = None

    return model


def init_simsiam_model(
    method: str,
    encoder: str,
    class_num: int,
    pred_head_features: int = 256,
    state_dict: Optional[dict] = None,
    freeze_encoder: bool = False,
    keep_enc_fc: bool = False,
    pred_head_structure: str = None
):
    if state_dict:
        # -- Infer model parameter used in the state_dict model
        num_features_first_fc_layer = state_dict[0]['fc.fc1.weight'].size()[0]
        enc_pred_dim = state_dict[0]['pred.fc2.weight'].size()[1]
        enc_feature_dim = state_dict[0]['fc.fc3.weight'].size()[0]
        # NOTE: strict=False is set to load PAWs and SimSiam models
        model = init_model(method, encoder, enc_feature_dim, enc_pred_dim)
        model.load_state_dict(state_dict[0], strict=True)
    else:
        enc_feature_dim = 4 * pred_head_features
        model = init_model(method, encoder, enc_feature_dim, pred_head_features)
        num_features_first_fc_layer = model.fc.fc1.in_features

    if freeze_encoder:
        for param in model.parameters():
            param.requires_grad = False

    if keep_enc_fc:
        model.pred = finetune_classifier(enc_feature_dim, class_num, pred_head_features, pred_head_structure)
    else:
        model.pred = None
        model.fc = finetune_classifier(num_features_first_fc_layer, class_num, pred_head_features, pred_head_structure)

    return model


def init_pytorch_model(pytorch_model: str, class_num: int):

    if (pytorch_model == 'vgg19'):
        model = models.vgg19()
        model.classifier[-1] = nn.Linear(in_features=4096, out_features=class_num, bias=True)
    elif (pytorch_model == 'squeezenet1_1'):
        model = models.squeezenet1_1()
        model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=class_num, kernel_size=(1, 1), stride=(1, 1))
    elif (pytorch_model == 'resnet50'):
        model = models.resnet50()
        model.fc = nn.Linear(in_features=2048, out_features=9, bias=True)
    elif (pytorch_model == 'alexnet'):
        model = models.alexnet()
        model.classifier[-1] = nn.Linear(in_features=4096, out_features=9, bias=True)
    else:
        raise NotImplementedError('Model not implemented, choose from [vgg19,squeezenet1_1,resnet50,alexnet]')
    return model


class Finetune(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.logger = logging.getLogger()  # Console log
        self.tblogger = TorchWriter()  # Tensorboard log

        # -- get state dict of a checkpoint if set
        state_dict = None
        if self.context.get_hparam('checkpoint_uuid'):
            checkpoint_path = Path('/checkpoints', self.context.get_hparam('checkpoint_uuid'), 'state_dict.pth')
            self.logger.info(f'-- loading checkpoint {checkpoint_path}')
            state_dict = torch.load(checkpoint_path, map_location="cpu")['models_state_dict']

        # -- Initialize model for specific method
        if self.context.get_hparam('method') == 'paws':
            model = init_paws_model(
                model_name=self.context.get_hparam('model_name'),
                output_dim=self.context.get_hparam('output_dim'),
                use_pred=self.context.get_hparam('use_pred_head'),
                number_classes=self.context.get_hparam('number_classes'),
                state_dict=state_dict,
                freeze_encoder=self.context.get_hparam('freeze_encoder'),
                dropout_rate=self.context.get_hparam('dropout_rate'),
                pred_head_structure=self.context.get_hparam('pred_head_structure'),
            )
        elif self.context.get_hparam('method') == 'simclr':
            model = init_simclr_model(
                encoder=self.context.get_hparam('encoder'),
                class_num=self.context.get_hparam('number_classes'),
                state_dict=state_dict,
                freeze_encoder=self.context.get_hparam('freeze_encoder'),
                keep_enc_fc=self.context.get_hparam('keep_enc_fc'),
                pred_head_structure=self.context.get_hparam('pred_head_structure'),
                pred_head_features=self.context.get_hparam('pred_head_features'),
            )
        elif self.context.get_hparam('method') == 'simsiam' or self.context.get_hparam('method') == 'simtriplet':
            model = init_simsiam_model(
                method=self.context.get_hparam('method'),
                encoder=self.context.get_hparam('encoder'),
                class_num=self.context.get_hparam('number_classes'),
                pred_head_features=self.context.get_hparam('pred_head_features'),
                state_dict=state_dict,
                freeze_encoder=self.context.get_hparam('freeze_encoder'),
                keep_enc_fc=self.context.get_hparam('keep_enc_fc'),
                pred_head_structure=self.context.get_hparam('pred_head_structure')
            )
        elif self.context.get_hparam('method') == 'pytorch_basic':
            model = init_pytorch_model(
                pytorch_model=self.context.get_hparam('pytorch_basic_model'),
                class_num=self.context.get_hparam('number_classes')
            )
        elif self.context.get_hparam('method') == 'pytorchvision':
            model = models.__dict__[self.context.get_hparam('encoder')](pretrained=True)
            fc_input_size = model.fc.in_features
            model.fc = finetune_classifier(
                fc_input_size,
                self.context.get_hparam('number_classes'),
                base_structure=self.context.get_hparam('pred_head_structure')
            )
        else:
            raise NotImplementedError('Method not implemented, choose from [paws, simclr, simsiam, simtriplet, pytorchvision]')

        # -- Log model infos
        self.logger.info(model)

        self.model = self.context.wrap_model(model)

        # -- Optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), self.context.get_hparam('lr'),
                                    momentum=self.context.get_hparam('momentum'),
                                    weight_decay=self.context.get_hparam('weight_decay'))
        if self.context.get_hparam('use_larc'):
            optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)
        self.optimizer = self.context.wrap_optimizer(optimizer)

        # -- Scheduler
        if self.context.get_hparam('scheduler'):
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.context.get_experiment_config()['searcher']['max_length']['batches'],
                verbose=False
            )
            self.scheduler = self.context.wrap_lr_scheduler(scheduler, LRScheduler.StepMode.STEP_EVERY_BATCH)

    def build_training_data_loader(self) -> DataLoader:
        augmentations = augs.__dict__[self.context.get_hparam('train_augmentation')](
            self.context.get_hparam('dataset'),
            self.context.get_hparam('normalize_data'),
            self.context.get_hparam('img_rescale_size')
        )

        train_set = init_data(
            self.context.get_hparam('dataset'),
            sub_set='train',
            split_size=self.context.get_hparam('split_size'),
            transform=augmentations
        )

        self.class_names = train_set.classes
        dataloader = DataLoader(
            train_set,
            batch_size=self.context.get_per_slot_batch_size(),
            num_workers=self.context.get_data_config()['worker'],
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )
        return dataloader

    def _build_valid_test_data_loader(self, subset='valid') -> DataLoader:
        # NOTE both valid and test data get augmented or not
        augmentations = augs.__dict__[self.context.get_hparam('test_augmentation')](
            self.context.get_hparam('dataset'),
            self.context.get_hparam('normalize_data'),
            self.context.get_hparam('img_rescale_size')
        )

        val_set = init_data(self.context.get_hparam('dataset'), sub_set=subset, transform=augmentations)
        dataloader = DataLoader(val_set,
                                batch_size=self.context.get_per_slot_batch_size(),
                                num_workers=self.context.get_data_config()['worker'],
                                pin_memory=True,
                                shuffle=False)
        return dataloader

    def build_validation_data_loader(self) -> DataLoader:
        return self._build_valid_test_data_loader('valid')

    def build_test_data_loader(self) -> DataLoader:
        return self._build_valid_test_data_loader('test')

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        # -- Encoder warm-up
        if type(self.context.get_hparam('freeze_encoder')) is int:
            if self.context.get_hparam('freeze_encoder') == batch_idx:
                for param in self.model.parameters():
                    param.requires_grad = True
                print('All params will be learned from now on!')

        # -- Train
        x, y = batch[0], batch[1]
        x = self.model(x)
        logits = nn.functional.log_softmax(x, dim=1)
        loss = nn.functional.nll_loss(logits, y)

        # -- Opt
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        # -- Metrics
        acc = metrics.accuracy(logits, y)

        return {'t_loss': loss, 't_acc': acc}

    def _calculate_outputs(self, dataloader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Uses self.model to predict outputs and labels
        Returns: A tuple (output, labels, preds, logits)
        """
        # -- Forward
        with torch.no_grad():
            outputs = []
            labels = []

            for batch in dataloader:
                x, y = batch[0], batch[1]
                x, y = self.context.to_device(x), self.context.to_device(y)

                x = self.model(x)

                outputs.append(x)
                labels.append(y)

        outputs = torch.cat(outputs)
        labels = torch.cat(labels)

        # -- Prediction
        logits = nn.functional.log_softmax(outputs, dim=1)
        preds = torch.argmax(logits, dim=1)

        return outputs, labels, preds, logits

    def _calculate_metrics(self, labels, preds, logits) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts with self.model labels for the given dataloader and calculates metrics for it
        Returns: A tuple (accuracy, f1, auroc, ece, loss)
        """
        # -- Metrics
        acc = metrics.accuracy(preds, labels)
        f1 = metrics.f1_score(logits, labels, average='macro', num_classes=self.context.get_hparam('number_classes'))
        auroc = metrics.auroc(logits, labels, num_classes=self.context.get_hparam('number_classes'))
        ece = metrics.calibration_error(torch.exp(logits), labels)
        loss = nn.functional.nll_loss(logits, labels)
        return acc, f1, auroc, ece, loss

    def _create_plots(self, outputs, preds, labels, batch_idx, subset):
        """
        Creates conf-matrix, umap and  tsne plots.
        """
        outputs = outputs.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        ident_string = f"batch:{batch_idx} ({subset}) - enc: {self.context.get_hparam('checkpoint_uuid')} - split: {self.context.get_hparam('split_size')}"

        self.tblogger.writer.add_figure(f'CONF-MATRIX_batch_{batch_idx}_{subset}',
                                        create_conf_matrix_plot(preds, labels, self.class_names,
                                                                title=f'Confusion Matrix ({ident_string})'),
                                        global_step=batch_idx
                                        )
        self.tblogger.writer.add_figure(f'UMAP_batch_{batch_idx}_{subset}',
                                        plot_UMAP(outputs, labels, title=f'UMAP ({ident_string})'),
                                        global_step=batch_idx
                                        )
        self.tblogger.writer.add_figure(f'TNSE_batch_{batch_idx}_{subset}',
                                        plot_TSNE(outputs, labels, title=f't-SNE ({ident_string})'),
                                        global_step=batch_idx
                                        )

    def evaluate_full_dataset(self, data_loader: DataLoader) -> Dict[str, Any]:
        batch_idx = self.context._current_batch_idx + 1

        # validation
        outputs, labels, preds, logits = self._calculate_outputs(data_loader)
        acc, f1, auroc, ece, loss = self._calculate_metrics(labels, preds, logits)
        stats = {'v_loss': loss, 'v_acc': acc, 'v_f1': f1, 'auroc': auroc, 'ece': ece}

        # -- Create tensorboard plots
        if batch_idx % int(self.context.get_hparam('plot_embedding_each')) == 0 or batch_idx == int(self.context.get_hparam('freeze_encoder')):
            self._create_plots(outputs, preds, labels, batch_idx, 'valid')

        # test
        max_length = self.context.get_experiment_config().get('searcher').get('max_length').get('batches')
        assert isinstance(max_length, int)

        if batch_idx == max_length:
            test_dataloader = self.build_test_data_loader()
            test_outputs, test_labels, test_preds, test_logits = self._calculate_outputs(test_dataloader)
            test_acc, test_f1, test_auroc, test_ece, test_loss = self._calculate_metrics(test_labels, test_preds, test_logits)

            stats['test_acc'] = test_acc
            stats['test_f1'] = test_f1
            stats['test_auroc'] = test_auroc
            stats['test_ece'] = test_ece
            stats['test_loss'] = test_loss

            # create test plots
            self._create_plots(test_outputs, test_preds, test_labels, batch_idx, 'test')

        print('fnished eval')
        return stats
