"""Loss functions for tumor segmentation training."""

import torch
import torch.nn as nn
from monai.losses import DiceLoss


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss for segmentation.

    Dice loss handles class imbalance (tumors are small relative to background).
    Cross-entropy provides stable gradients early in training.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        sigmoid: bool = True,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
    ):
        super().__init__()

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        self.dice = DiceLoss(
            sigmoid=sigmoid,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            to_onehot_y=False,
        )
        self.ce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice + CE loss.

        Args:
            pred: Predicted logits [B, C, H, W, D]
            target: Ground truth labels [B, C, H, W, D] (one-hot) or [B, H, W, D] (integer)

        Returns:
            Scalar loss value.
        """
        dice_loss = self.dice(pred, target)
        ce_loss = self.ce(pred, target.float())

        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


class DeepSupervisionLoss(nn.Module):
    """Weighted loss over deep supervision outputs.

    Assigns decreasing weights to coarser-scale predictions.
    """

    def __init__(self, base_loss: nn.Module, weights: list[float] | None = None):
        super().__init__()

        self.base_loss = base_loss
        self.weights = weights

    def forward(
        self,
        predictions: list[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted sum of losses at each supervision scale.

        Args:
            predictions: List of predictions at different scales
            target: Ground truth (will be interpolated to match each scale)

        Returns:
            Scalar loss value.
        """
        if self.weights is None:
            n = len(predictions)
            weights = [1.0 / (2**i) for i in range(n)]
            total = sum(weights)
            weights = [w / total for w in weights]
        else:
            weights = self.weights

        total_loss = torch.tensor(0.0, device=predictions[0].device)

        for pred, weight in zip(predictions, weights):
            total_loss = total_loss + weight * self.base_loss(pred, target)

        return total_loss
