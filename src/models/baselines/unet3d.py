"""3D U-Net baseline (Cicek et al., 2016)."""

import torch.nn as nn
from monai.networks.nets import UNet


class UNet3D(nn.Module):
    """Standard 3D U-Net baseline using MONAI implementation."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        features: tuple[int, ...] = (32, 64, 128, 256, 512),
    ):
        super().__init__()

        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            channels=features,
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="instance",
        )

    def forward(self, x):
        return {"pred": self.model(x)}
