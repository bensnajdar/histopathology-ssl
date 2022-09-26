import torch
from torch import nn
# noinspection PyProtectedMember
from cbmi_utils.pytorch.models.wide_resnet import WideResNet


class SimClrWideResNet(WideResNet):
    def __init__(
            self,
            feature_dim: int = 2048,
            encoder: str = 'wide_resnet28w2'
    ):
        enc_depth, enc_widen_factor = globals()[encoder]()
        super().__init__(enc_depth, enc_widen_factor)
        self.enc_widen_factor = enc_widen_factor

        prev_dim = self.get_pre_fc_dim()
        self.fc = nn.Sequential(
            nn.Linear(prev_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )
        self.pred = None

    def get_pre_fc_dim(self):
        return 64 * self.enc_widen_factor

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self._forward_backbone(x)
        return self._forward_head(x)


def wide_resnet28w2():
    return 28, 2
