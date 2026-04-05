"""UNETR baseline (Hatamizadeh et al., 2022)."""

import torch.nn as nn
from monai.networks.nets import UNETR as _MonaiUNETR


class UNETR(nn.Module):
    """UNETR baseline using MONAI implementation."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        img_size: tuple[int, ...] = (128, 128, 128),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
    ):
        super().__init__()

        self.model = _MonaiUNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            norm_name="instance",
            spatial_dims=3,
        )

    def forward(self, x):
        return {"pred": self.model(x)}
