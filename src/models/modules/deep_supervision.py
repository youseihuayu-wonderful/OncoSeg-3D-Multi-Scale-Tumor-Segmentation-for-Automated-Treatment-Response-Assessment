"""Deep Supervision heads for multi-scale loss computation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSupervisionHead(nn.Module):
    """Auxiliary segmentation heads at intermediate decoder stages.

    During training, each intermediate decoder feature map gets its own
    classification head. The final loss is a weighted sum of all heads,
    which stabilizes training and improves small tumor detection.
    """

    def __init__(self, encoder_dims: list[int], num_classes: int):
        super().__init__()

        self.heads = nn.ModuleList(
            [nn.Conv3d(dim, num_classes, kernel_size=1) for dim in encoder_dims]
        )

    def forward(self, intermediates: list[torch.Tensor]) -> list[torch.Tensor]:
        """Produce auxiliary segmentation maps at each decoder scale.

        Args:
            intermediates: List of intermediate decoder features

        Returns:
            List of segmentation logits, each upsampled to match the finest resolution.
        """
        outputs = []
        target_size = intermediates[-1].shape[2:]  # Finest resolution spatial dims

        for feat, head in zip(intermediates, self.heads):
            logits = head(feat)
            if logits.shape[2:] != target_size:
                logits = F.interpolate(
                    logits, size=target_size, mode="trilinear", align_corners=False
                )
            outputs.append(logits)

        return outputs
