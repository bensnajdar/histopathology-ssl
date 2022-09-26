import logging
import torch
import torch.nn as nn
import src.resnet as resnet
import src.densenet as densenet
import src.wide_resnet as wide_resnet
import torchvision.transforms as transforms
import torchmetrics.functional.classification as metrics

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, Optional
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader, TorchData, LRScheduler
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from src.data_manager_rework import init_data, make_transforms, make_multicrop_transform, make_labels_matrix
from src.optimizer import init_opt
from src.loss import init_paws_loss
from src.utils import TSNE_vis, UMAP_vis, create_conf_matrix_plot
from src.init import init_data_cbmi_utils, get_class_names
from torch.optim.lr_scheduler import CosineAnnealingLR

device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_model(model_name='resnet50', output_dim=128, pretrained_base_model_path=None, pretrained_paws_model_path=None,
               use_pred=False, number_classes=2):
    """Initialize a ResNet of WideResNet model

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
        encoder = wide_resnet.__dict__[model_name](dropout_rate=0.0)
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
    if (pretrained_base_model_path == 'None'):
        pretrained_base_model_path = None
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
    if (pretrained_paws_model_path == 'None'):
        pretrained_paws_model_path = None
    if pretrained_paws_model_path:
        state_dict = torch.load(pretrained_paws_model_path)['models_state_dict']
        encoder.load_state_dict(state_dict[0])

    # add prediction head for finetuning downstream tasks
    if use_pred:
        encoder.pred = torch.nn.Sequential(
            OrderedDict([
                ('fc1', torch.nn.Linear(output_dim, output_dim // 2)),
                ('bn1', torch.nn.BatchNorm1d(output_dim // 2)),
                ('relu1', torch.nn.ReLU(inplace=True)),
                ('fc2', torch.nn.Linear(output_dim // 2, output_dim // 4)),
                ('bn2', torch.nn.BatchNorm1d(output_dim // 4)),
                ('relu2', torch.nn.ReLU(inplace=True)),
                ('fc3', torch.nn.Linear(output_dim // 4, number_classes))
            ])
        )

    return encoder


class Paws_Encoder(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.logger = logging.getLogger()  # Console log
        self.tblogger = TorchWriter()  # Tensorboard log

        # -- init model from determined config
        encoder = init_model(
            model_name=self.context.get_hparam('model_name'),
            output_dim=self.context.get_hparam('output_dim'),
            pretrained_base_model_path=self.context.get_hparam('pretrained_base_model_path'),
            pretrained_paws_model_path=self.context.get_hparam('pretrained_paws_model_path'),
        )

        if (self.context.get_hparam('multicrop') > 0):
            crop_scale = (self.context.get_hparam('crop_scale_min'), self.context.get_hparam('crop_scale_max'))
        else:
            crop_scale = (0.5, 1.0)

        mc_scale = (self.context.get_hparam('mc_scale_min'), self.context.get_hparam('mc_scale_max'))
        mc_size = self.context.get_hparam('mc_resize_resolution')

        # -- create data transformations
        transform = make_transforms(
            dataset_name=self.context.get_hparam('dataset'),
            std_resize_resolution=self.context.get_hparam('std_resize_resolution'),
            crop_scale=crop_scale,
            basic_augmentations=False,
            color_jitter=self.context.get_hparam('color_jitter_strength'),
            normalize=self.context.get_hparam('normalize'),
        )

        multicrop_transform = (self.context.get_hparam('multicrop'), None)
        if self.context.get_hparam('multicrop') > 0:
            multicrop_transform = make_multicrop_transform(
                dataset_name=self.context.get_hparam('dataset'),
                num_crops=self.context.get_hparam('multicrop'),
                size=mc_size,
                crop_scale=mc_scale,
                normalize=self.context.get_hparam('normalize'),
                color_distortion=self.context.get_hparam('color_jitter_strength')
            )

        unsupervised_loader, unsupervised_sampler, supervised_loader, supervised_sampler = init_data(
            dataset_name=self.context.get_hparam('dataset'),
            split_size=self.context.get_hparam('split_size'),
            split_seed=self.context.get_hparam('split_seed'),
            split_file=self.context.get_hparam('split_file'),
            transform=transform,
            supervised_views=self.context.get_hparam('supervised_views'),
            u_batch_size=self.context.get_hparam('unsupervised_batch_size'),
            s_batch_size=self.context.get_hparam('supervised_imgs_per_class'),
            unique_classes=self.context.get_hparam('unique_classes_per_rank'),
            classes_per_batch=self.context.get_hparam('classes_per_batch'),
            multicrop_transform=multicrop_transform,
            world_size=1,
            rank=0,
            sub_set='train'
        )

        self.unsupervised_loader = unsupervised_loader
        self.unsupervised_sampler = unsupervised_sampler
        self.supervised_loader = supervised_loader
        self.supervised_sampler = supervised_sampler

        # NOTE: This only works if the support set data_loader uses the `ClassStratifiedSampler` as sampler
        # -- create label for the support set
        self.labels_matrix = make_labels_matrix(
            num_classes=self.context.get_hparam('classes_per_batch'),
            s_batch_size=self.context.get_hparam('supervised_imgs_per_class'),
            device=device_,
            unique_classes=self.context.get_hparam('unique_classes_per_rank'),
            smoothing=self.context.get_hparam('label_smoothing')
        )
        ipe = len(unsupervised_loader)
        ipe_super = len(supervised_loader)
        self.logger.info(f'iterations per epoch: {ipe}')
        self.logger.info(f'iterations per epoch super: {ipe_super}')

        # -- init optimizer and scheduler
        encoder, optimizer, scheduler = init_opt(
            encoder=encoder,
            weight_decay=self.context.get_hparam('weight_decay'),
            start_lr=self.context.get_hparam('start_lr'),
            ref_lr=self.context.get_hparam('lr'),
            final_lr=self.context.get_hparam('final_lr'),
            ref_mom=self.context.get_hparam('momentum'),
            nesterov=self.context.get_hparam('nesterov'),
            iterations_per_epoch=ipe,
            warmup=self.context.get_hparam('warmup'),
            batches=self.context.get_experiment_config()["searcher"]["max_length"]["batches"]
        )
        # -- init grad scaler
        scaler = torch.cuda.amp.GradScaler(enabled=self.context.get_hparam('use_fp16'))

        # Do determined wrapping
        self.scaler = self.context.wrap_scaler(scaler)
        self.model = self.context.wrap_model(encoder)
        self.optimizer = self.context.wrap_optimizer(optimizer)
        self.scheduler = scheduler

        # Loss
        self.paws = init_paws_loss(
            multicrop=self.context.get_hparam('multicrop'),
            tau=self.context.get_hparam('temperature'),
            T=self.context.get_hparam('sharpen'),
            me_max=self.context.get_hparam('me_max')
        )

    def build_training_data_loader(self) -> DataLoader:
        return self.unsupervised_loader

    def build_validation_data_loader(self) -> DataLoader:
        # Get validation dataloader for evaluation
        transform = make_transforms(
            dataset_name=self.context.get_hparam('dataset'),
            std_resize_resolution=self.context.get_hparam('std_resize_resolution'),
            basic_augmentations=True,
            normalize=self.context.get_hparam('normalize')
        )

        data_loader, dist_sampler = init_data(
            finetune=True,
            dataset_name=self.context.get_hparam('dataset'),
            transform=transform,
            u_batch_size=None,
            s_batch_size=256,
            split_size=0.98,
            split_seed=42,
            split_file=False,
            classes_per_batch=None,
            world_size=1,
            rank=0,
            sub_set='valid'
        )

        self.val_dist_sampler = dist_sampler
        return data_loader

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        self.unsupervised_sampler.set_epoch(epoch_idx)
        if self.supervised_sampler is not None:
            self.supervised_sampler.set_epoch(epoch_idx)

        def load_imgs():
            device = self.context.device
            uimgs = [u.to(device, non_blocking=True) for u in batch[:-1]]
            # -- supervised imgs
            global iter_supervised
            try:
                sdata = next(iter_supervised)
            except Exception:
                iter_supervised = iter(self.supervised_loader)
                sdata = next(iter_supervised)
            finally:
                labels = torch.cat([self.labels_matrix for _ in range(self.context.get_hparam('supervised_views'))])
                simgs = [s.to(device, non_blocking=True) for s in sdata[:-1]]
            # -- concatenate supervised imgs and unsupervised imgs
            imgs = simgs + uimgs
            return imgs, labels

        batch, labels = load_imgs()

        with torch.cuda.amp.autocast(enabled=self.context.get_hparam('use_fp16')):
            h, z = self.model(batch, return_before_head=True)

        # Compute paws loss in full precision
        with torch.cuda.amp.autocast(enabled=False):

            # Step 1. convert representations to fp32
            h, z = h.float(), z.float()

            # Step 2. determine anchor views/supports and their
            #         corresponding target views/supports
            num_support = self.context.get_hparam('supervised_views') * self.context.get_hparam('supervised_imgs_per_class') * self.context.get_hparam('classes_per_batch')

            # -- NOTE: Z and H are actually the same mapping if the network has no model.pred (drediction head)
            anchor_supports = z[:num_support]
            anchor_views = z[num_support:]

            target_supports = h[:num_support].detach()
            target_views = h[num_support:].detach()
            target_views = torch.cat([target_views[self.context.get_hparam('unsupervised_batch_size'):2*self.context.get_hparam('unsupervised_batch_size')],
                                      target_views[:self.context.get_hparam('unsupervised_batch_size')]], dim=0)

            # Step 3. compute paws loss with me-max regularization
            ploss, me_max, anchor_probs_max, target_max_mean, target_sharp_max_mean = self.paws(
                anchor_views=anchor_views,
                anchor_supports=anchor_supports,
                anchor_support_labels=labels,
                target_views=target_views,
                target_supports=target_supports,
                target_support_labels=labels
            )
            loss = ploss + me_max

        # Backprop and update
        self.context.backward(self.scaler.scale(loss))
        self.context.step_optimizer(self.optimizer, scaler=self.scaler)
        self.scaler.update()
        self.scheduler.step()

        idx = 0
        if batch_idx % 195 == 0:
            self.logger.info('Created Histogram of Max Probs of the Anchor View')
            self.tblogger.writer.add_histogram('max probs histo', anchor_probs_max + idx, idx)
            idx += 1

        metrics = {}
        for i, learning_rate in enumerate(self.scheduler.get_last_lr()):
            metrics[f'LR_Param_{i}'] = learning_rate

        metrics.update({
            'Loss': loss.item(),
            'PLoss': ploss.item(),
            'RLoss': me_max.item(),
            'MeanMaxProbs': torch.mean(anchor_probs_max).item(),
            'MinOfMaxProbs': torch.min(anchor_probs_max).item(),
            'MeanMaxProbsTarget': target_max_mean.item(),
            'MeanMaxProbsTargetSharp': target_sharp_max_mean.item()
        })

        self.t_loss = loss.item()
        return metrics

    def evaluate_full_dataset(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        batch_idx = self.context._current_batch_idx + 1

        # Create UMAP plots for tensorboard each x batches
        if batch_idx % int(self.context.get_hparam('plot_embedding_each')) == 0:
            with torch.no_grad():
                outputs = []
                labels = []

                for batch in data_loader:
                    x, label = self.context.to_device(batch[0]), batch[1]
                    z = self.model(x, return_before_head=False)
                    outputs.append(z.detach().cpu())
                    labels.append(label)

                outputs = torch.cat(outputs)
                labels = torch.cat(labels)
                outputs = outputs.numpy()
                labels = labels.numpy()

                self.logger.info(
                    f'Labels shape: {labels.shape}'
                )
                self.tblogger.writer.add_figure(
                    f'encoder_space_UMAP_valid_batch_{batch_idx}',
                    UMAP_vis(outputs, labels, set_name=self.context.get_hparam('dataset')),
                    global_step=batch_idx
                )
                self.tblogger.writer.add_figure(
                    f'encoder_space_TNSE_valid_batch_{batch_idx}',
                    TSNE_vis(outputs, labels, set_name=self.context.get_hparam('dataset')),
                    global_step=batch_idx
                )
        return {"t_loss": self.t_loss}


class Paws_Fintenue(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.logger = logging.getLogger()  # Console log
        self.tblogger = TorchWriter()  # Tensorboard log

        # -- init model from determined config
        if self.context.get_hparam('checkpoint_uuid'):
            pretrained_paws_model_path_ = Path('/checkpoints', self.context.get_hparam('checkpoint_uuid'), 'state_dict.pth')
        else:
            pretrained_paws_model_path_ = 'None'

        model = init_model(
            model_name=self.context.get_hparam('model_name'),
            output_dim=self.context.get_hparam('output_dim'),
            use_pred=True,
            number_classes=self.context.get_hparam('number_classes'),
            pretrained_paws_model_path=pretrained_paws_model_path_,
        )

        if self.context.get_hparam('freeze_paws_encoder') is not False:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.pred.parameters():
                param.requires_grad = True  # uses only prediction head for finetuning

        self.model = self.context.wrap_model(model)

        # -- Optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(), self.context.get_hparam('lr'),
            momentum=self.context.get_hparam('momentum'),
            weight_decay=self.context.get_hparam('weight_decay')
        )
        self.optimizer = self.context.wrap_optimizer(optimizer)

        # -- Scheduler
        if self.context.get_hparam('scheduler'):
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.context.get_experiment_config()['searcher']['max_length']['batches'],
                verbose=False
            )
            self.scheduler = self.context.wrap_lr_scheduler(scheduler, LRScheduler.StepMode.STEP_EVERY_BATCH)

    def _get_augmentations(self, img_size: Optional[int] = None):
        augmentations = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])
        return augmentations

    def build_training_data_loader(self) -> DataLoader:
        train_set = init_data_cbmi_utils(
            self.context.get_hparam('dataset'),
            sub_set='train',
            split_size=self.context.get_hparam('split_size'),
            transform=self._get_augmentations()
        )

        dataloader = DataLoader(
            train_set,
            batch_size=self.context.get_per_slot_batch_size(),
            num_workers=self.context.get_data_config()['worker'],
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )
        return dataloader

    def build_validation_data_loader(self) -> DataLoader:
        val_set = init_data_cbmi_utils(
            self.context.get_hparam('dataset'),
            sub_set='valid',
            transform=self._get_augmentations()
        )
        dataloader = DataLoader(
            val_set,
            batch_size=self.context.get_per_slot_batch_size(),
            num_workers=self.context.get_data_config()['worker'],
            pin_memory=True,
            shuffle=False
        )

        return dataloader

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        # -- Encoder warm-up
        if type(self.context.get_hparam('freeze_paws_encoder')) is int:
            if self.context.get_hparam('freeze_paws_encoder') == batch_idx:
                for param in self.model.parameters():
                    param.requires_grad = True
                print('All params will be learned from now on!')

        # -- Train
        x, y = batch[0], batch[1]
        x = self.model(x)
        logits = nn.functional.log_softmax(x, dim=1)
        loss = nn.functional.nll_loss(logits, y)

        # -- Metrics
        preds = torch.argmax(logits, dim=1)
        acc = metrics.accuracy(preds, y)

        # -- Opt
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        return {'t_loss': loss, 't_acc': acc}

    def evaluate_full_dataset(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        # -- Forward
        with torch.no_grad():
            # Output container first dim represents the output sequence p1, p2, z1 or z2
            outputs = []
            labels = []

            for batch in data_loader:
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

        # -- Metrics
        acc = metrics.accuracy(preds, labels)
        auroc = metrics.auroc(logits, labels, num_classes=self.context.get_hparam('number_classes'))
        ece = metrics.calibration_error(torch.exp(logits), labels)
        loss = nn.functional.nll_loss(logits, labels)

        # -- Create tensorboard plots
        batch_idx = self.context._current_batch_idx + 1
        if batch_idx % int(self.context.get_hparam('plot_embedding_each')) == 0 or batch_idx == int(self.context.get_hparam('freeze_paws_encoder')):
            # -- Convert
            outputs = outputs.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            self.tblogger.writer.add_figure(
                f'CONF-MATRIX_batch_{batch_idx}',
                create_conf_matrix_plot(
                    preds,
                    labels,
                    get_class_names(self.context.get_hparam('dataset'))),
                global_step=batch_idx
            )

            self.tblogger.writer.add_figure(
                f'UMAP_batch_{batch_idx}',
                UMAP_vis(outputs, labels, set_name=self.context.get_hparam('dataset')),
                global_step=batch_idx
            )
            self.tblogger.writer.add_figure(
                f'TNSE_batch_{batch_idx}',
                TSNE_vis(outputs, labels, set_name=self.context.get_hparam('dataset')),
                global_step=batch_idx
            )
        return {'v_loss': loss, 'v_acc': acc, 'auroc': auroc, 'ece': ece}
