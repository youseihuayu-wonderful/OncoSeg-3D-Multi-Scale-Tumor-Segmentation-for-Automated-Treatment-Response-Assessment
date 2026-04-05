"""Swin UNETR baseline (Tang et al., 2022)."""

import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETRBaseline(nn.Module):
    """Swin UNETR baseline using MONAI implementation."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        feature_size: int = 48,
        depths: tuple[int, ...] = (2, 2, 2, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        **kwargs,
    ):
        super().__init__()

        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            norm_name="instance",
            spatial_dims=3,
        )

    def forward(self, x):
        return {"pred": self.model(x)}
