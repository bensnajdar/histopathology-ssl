from typing import Optional, Tuple, Callable
import torch
import torch.nn as nn
from .simsiam import SimSiamResNetPAWs, SimSiamPT, SimSiamWideResNetPAWs


class SimTriplet(SimSiamResNetPAWs):
    def __init__(
        self,
        feature_dim: int = 2048,
        pred_dim: Optional[int] = None,
        encoder: str = 'resnet50'
    ):
        super().__init__(feature_dim, pred_dim, encoder)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None, x3: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Creates encoder and prediction mappings. Encoder projections are detached from the computational graph, see
            https://arxiv.org/abs/2011.10566 for detailed notations.

        Args:
            x1 (torch.Tensor): first image views
            x2 (torch.Tensor): second image views

        Returns:
            (torch.Tensor):  p1, p2, z1, z2: encoder and predictor mappings
        """

        # -- Note if only one input is given, we expect a finetune and do a normal forward
        if x2 is None and x3 is None:
            x = self._forward_backbone(x1)
            return self._forward_head(x)

        z1 = self._forward_backbone(x1)
        z2 = self._forward_backbone(x2)
        z3 = self._forward_backbone(x3)

        p1 = self._forward_head(z1)
        p2 = self._forward_head(z2)
        p3 = self._forward_head(z3)

        return p1, p2, p3, z1.detach(), z2.detach(), z3.detach()


class SimTripWideResNetPAWs(SimSiamWideResNetPAWs):
    def __init__(
        self,
        feature_dim: int = 512,
        pred_dim: Optional[int] = None,
        encoder: str = 'wide_resnet28w2'
    ):
        super().__init__(feature_dim, pred_dim, encoder)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None, x3: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Creates encoder and prediction mappings. Encoder projections are detached from the computational graph, see
            https://arxiv.org/abs/2011.10566 for detailed notations.

        Args:
            x1 (torch.Tensor): first image views
            x2 (torch.Tensor): second image views
            x3 (torch.Tensor): third image views

        Returns:
            (torch.Tensor):  p1, p2, p3, z1, z2, z3: encoder and predictor mappings
        """

        # -- Note if only one input is given, we expect a finetune and do a normal forward
        if x2 is None and x3 is None:
            x = self._forward_backbone(x1)
            return self._forward_head(x)

        z1 = self._forward_backbone(x1)
        z2 = self._forward_backbone(x2)
        z3 = self._forward_backbone(x3)

        p1 = self._forward_head(z1)
        p2 = self._forward_head(z2)
        p3 = self._forward_head(z3)

        return p1, p2, p3, z1.detach(), z2.detach(), z3.detach()


class SimTripletPT(SimSiamPT):
    def __init__(
        self,
        base_encoder: str = 'resnet50',
        feature_dim: int = 2048,
        pred_dim: Optional[int] = None
    ):
        super().__init__(base_encoder, feature_dim, pred_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None, x3: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Creates encoder and prediction mappings. Encoder projections are detached from the computational graph, see
            https://arxiv.org/abs/2011.10566 for detailed notations.

        Args:
            x1 (torch.Tensor): first image views
            x2 (torch.Tensor): second image views

        Returns:
            (torch.Tensor):  p1, p2, z1, z2: encoder and predictor mappings
        """

        # -- Note if only one input is given, we expect a finetune and do a normal forward
        if x2 is None and x3 is None:
            return self.fc(self.encoder(x1))

        z1 = self.fc(self.encoder(x1))
        z2 = self.fc(self.encoder(x2))
        z3 = self.fc(self.encoder(x3))

        p1 = self._forward_pred(z1)
        p2 = self._forward_pred(z2)
        p3 = self._forward_pred(z3)

        return p1, p2, p3, z1.detach(), z2.detach(), z3.detach()


class SimTripletLoss(nn.Module):
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

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:

        d1 = self.distance(p1, z2).mean()
        d2 = self.distance(p2, z1).mean()
        d3 = self.distance(p1, z3).mean()
        d4 = self.distance(p3, z1).mean()

        return -(torch.sum(torch.stack([d1, d2, d3, d4]))) * 0.5
