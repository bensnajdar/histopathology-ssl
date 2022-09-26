from typing import Callable, List, Tuple, Optional
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from cbmi_utils.pytorch.models.wide_resnet import WideResNet
from cbmi_utils.pytorch.models.paws_resnet import ResNet, Bottleneck, BasicBlock


def finetune_classifier(input_size, class_num, hidden_feat=128, base_structure: str = None):
    if base_structure == 'three_layer_mlp':
        pred_head = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_feat // 2)),
            ('bn1', nn.BatchNorm1d(hidden_feat // 2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(hidden_feat // 2, hidden_feat // 4)),
            ('bn2', nn.BatchNorm1d(hidden_feat // 4)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(hidden_feat // 4, class_num))
        ]))
    elif base_structure == 'two_layer_mlp':
        pred_head = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_feat // 2)),
            ('bn1', nn.BatchNorm1d(hidden_feat // 2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(hidden_feat // 2, class_num))
        ]))
    elif base_structure == 'one_layer_mlp':
        pred_head = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, class_num))
        ]))
    else:
        pred_head = nn.Sequential(
            nn.Linear(input_size, hidden_feat, bias=False),
            nn.BatchNorm1d(hidden_feat),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feat, hidden_feat, bias=False),
            nn.BatchNorm1d(hidden_feat),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feat, class_num)
        )

    return pred_head


def _predictor(feature_dim=2048, hidden_dim=512):
    return nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(feature_dim, hidden_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(hidden_dim)),
            ('re1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(hidden_dim, feature_dim))
        ])
    )


def _encoder_fc(feature_dim=2048, enc_dim=512):
    return nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(enc_dim, enc_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(enc_dim)),
            ('re1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(enc_dim, enc_dim, bias=False)),
            ('bn2', nn.BatchNorm1d(enc_dim)),
            ('re2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(enc_dim, feature_dim, bias=False)),
            # HACK: Removed to simulate PAWs FC - older Checkpoints won't work that way
            # ('bn3', nn.BatchNorm1d(feature_dim, affine=False))
        ])
    )


class SimSiamPT(nn.Module):
    def __init__(self, base_encoder, feature_dim: int = 2048, pred_dim: Optional[int] = 128) -> None:
        super(SimSiamPT, self).__init__()

        # -- initial a pytorchvision network
        self.encoder = models.__dict__[base_encoder]()

        # -- base_encoder specific overriding of the last layer
        if 'res' in base_encoder:
            prev_dim = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Identity()
        elif 'efficient' in base_encoder:
            prev_dim = self.encoder.classifier[-1].in_features
            self.encoder.classifier = nn.Identity()
        else:
            raise NotImplementedError(f'No implementation for such base_encoder {base_encoder}. Implement replacement of the fully connected layers.')

        self.fc = _encoder_fc(feature_dim=feature_dim, enc_dim=prev_dim)

        # Predictor
        if pred_dim is not None:
            self.pred = _predictor(feature_dim=feature_dim, hidden_dim=pred_dim)

    def _forward_pred(self, x):
        if self.pred is not None:
            x = self.pred(x)
        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Creates encoder and prediction mappings. Encoder projections are detached from the computational graph, see
            https://arxiv.org/abs/2011.10566 for detailed notations.

        Args:
            x1 (torch.Tensor): first image views
            x2 (torch.Tensor): second image views

        Returns:
            (torch.Tensor):  p1, p2, z1, z2: encoder and predictor mappings
        """
        if x2 is None:
            return self.fc(self.encoder(x1))

        z1 = self.fc(self.encoder(x1))
        z2 = self.fc(self.encoder(x2))

        p1 = self._forward_pred(z1)
        p2 = self._forward_pred(z2)

        return p1, p2, z1.detach(), z2.detach()


class SimSiamWideResNetPAWs(WideResNet):
    def __init__(
        self,
        feature_dim: int = 2048,
        pred_dim: Optional[int] = None,
        encoder: str = 'wide_resnet28w2'
    ):
        enc_depth, enc_widen_factor = globals()[encoder]()
        super().__init__(enc_depth, enc_widen_factor)

        prev_dim = 64 * enc_widen_factor
        self.fc = _encoder_fc(feature_dim=feature_dim, enc_dim=prev_dim)

        if pred_dim is not None:
            self.pred = _predictor(feature_dim=feature_dim, hidden_dim=pred_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Creates encoder and prediction mappings. Encoder projections are detached from the computational graph, see
            https://arxiv.org/abs/2011.10566 for detailed notations.

        Args:
            x1 (torch.Tensor): first image views
            x2 (torch.Tensor): second image views

        Returns:
            (torch.Tensor):  p1, p2, z1, z2: encoder and predictor mappings
        """

        # -- Note if only one input is given, we expect a finetune and do a normal forward
        if x2 is None:
            x = self._forward_backbone(x1)
            return self._forward_head(x)

        z1 = self._forward_backbone(x1)
        z2 = self._forward_backbone(x2)

        p1 = self._forward_head(z1)
        p2 = self._forward_head(z2)

        return p1, p2, z1.detach(), z2.detach()


class SimSiamResNetPAWs(ResNet):
    def __init__(
        self,
        feature_dim: int = 2048,
        pred_dim: Optional[int] = None,
        encoder: str = 'resnet50'
    ):
        block, layers, enc_widen_factor = globals()[encoder]()
        super().__init__(block, layers, zero_init_residual=True, widen=enc_widen_factor)

        if block == BasicBlock:
            prev_dim = self.layer4[-1].conv2.weight.size()[0]
        else:
            prev_dim = self.layer4[-1].conv3.weight.size()[0]

        self.fc = _encoder_fc(feature_dim=feature_dim, enc_dim=prev_dim)

        if pred_dim is not None:
            self.pred = _predictor(feature_dim=feature_dim, hidden_dim=pred_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Creates encoder and prediction mappings. Encoder projections are detached from the computational graph, see
            https://arxiv.org/abs/2011.10566 for detailed notations.

        Args:
            x1 (torch.Tensor): first image views
            x2 (torch.Tensor): second image views

        Returns:
            (torch.Tensor):  p1, p2, z1, z2: encoder and predictor mappings
        """

        # -- Note if only one input is given, we expect a finetune and do a normal forward
        if x2 is None:
            x = self._forward_backbone(x1)
            return self._forward_head(x)

        z1 = self._forward_backbone(x1)
        z2 = self._forward_backbone(x2)

        p1 = self._forward_head(z1)
        p2 = self._forward_head(z2)

        return p1, p2, z1.detach(), z2.detach()


class SimSiamLoss(nn.Module):
    """ Implements loss with a given distance D:
        L = \frac{1}{2} D(p_1,z_2) + \frac{1}{2} D(p_2,z_1)

    Args:
        distance_metric (Callable[[torch.Tensor], torch.Tensor]): A function that calculates a distance metric between two tensor
    """
    def __init__(self, distance_metric: Callable[[torch.Tensor], torch.Tensor] = None) -> None:
        super().__init__()

        if distance_metric:
            self.distance = distance_metric
        else:
            self.distance = nn.CosineSimilarity(dim=1)

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return -(self.distance(p1, z2).mean() + self.distance(p2, z1).mean()) * 0.5


class TwoCropsTransform:
    """ Applies a given transformation set to a input twice
    """
    def __init__(self, base_transform: torchvision.transforms.Compose) -> None:
        self.base_transform = base_transform

    def __call__(self, x) -> List[torch.Tensor]:
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


# ResNets
def resnet18():
    return BasicBlock, [2, 2, 2, 2], 1


def resnet34():
    return BasicBlock, [3, 4, 6, 3], 1


def resnet50():
    return Bottleneck, [3, 4, 6, 3], 1


def resnet50w2():
    return Bottleneck, [3, 4, 6, 3], 2


def resnet50w4():
    return Bottleneck, [3, 4, 6, 3], 4


def resnet101():
    return Bottleneck, [3, 4, 23, 3], 1


def resnet101w2():
    return Bottleneck, [3, 4, 23, 3], 2


def resnet151():
    return Bottleneck, [3, 8, 36, 3], 1


def resnet151w2():
    return Bottleneck, [3, 8, 36, 3], 2


def resnet200():
    return Bottleneck, [3, 24, 36, 3], 1


# Wide ResNets
def wide_resnet10w2():
    return 10, 2


def wide_resnet16w2():
    return 16, 2


def wide_resnet16w4():
    return 16, 4


def wide_resnet22w2():
    return 22, 2


def wide_resnet22w4():
    return 22, 4


def wide_resnet28w2():
    return 28, 2


def wide_resnet28w4():
    return 28, 4


def wide_resnet28w8():
    return 28, 8


def wide_resnet34w2():
    return 34, 2


def wide_resnet34w4():
    return 34, 4


def wide_resnet34w8():
    return 34, 8


def wide_resnet40w2():
    return 40, 2
