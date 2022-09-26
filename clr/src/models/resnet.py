import torch
from torch import nn
# noinspection PyProtectedMember
from cbmi_utils.pytorch.models.paws_resnet import ResNet, Bottleneck, BasicBlock


class SimClrResNet(ResNet):
    def __init__(
            self,
            feature_dim: int = 2048,
            encoder: str = 'resnet50'
    ):
        block, layers, enc_widen_factor = globals()[encoder]()
        super().__init__(block, layers, zero_init_residual=True, widen=enc_widen_factor)
        self.encoder = encoder

        prev_dim = self.get_pre_fc_dim()

        self.fc = nn.Sequential(
            nn.Linear(prev_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )
        self.pred = None

    def get_pre_fc_dim(self):
        block, _, _ = globals()[self.encoder]()
        if block == BasicBlock:
            return self.layer4[-1].conv2.weight.size()[0]
        else:
            return self.layer4[-1].conv3.weight.size()[0]

    def forward(self, x1: torch.Tensor, **_kwargs) -> torch.Tensor:
        """ Creates encoder embeddings for images.

        Args:
            x1 (torch.Tensor): image view
            **kwargs: should be empty
        """
        assert not _kwargs, 'got further arguments, which is not allowed'
        x = self._forward_backbone(x1)
        return self._forward_head(x)


def resnet18():
    return BasicBlock, [2, 2, 2, 2], 1


def resnet50():
    return Bottleneck, [3, 4, 6, 3], 1

