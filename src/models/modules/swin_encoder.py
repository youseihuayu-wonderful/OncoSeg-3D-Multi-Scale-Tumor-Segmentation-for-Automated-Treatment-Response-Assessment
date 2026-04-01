"""3D Swin Transformer Encoder for volumetric medical image feature extraction."""

import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer


class SwinEncoder3D(nn.Module):
    """3D Swin Transformer encoder with hierarchical feature extraction.

    Uses MONAI's SwinTransformer implementation as the backbone, producing
    multi-scale features at 4 resolution stages.

    Output feature maps:
        Stage 1: [B, embed_dim,    H/4,  W/4,  D/4]
        Stage 2: [B, embed_dim*2,  H/8,  W/8,  D/8]
        Stage 3: [B, embed_dim*4,  H/16, W/16, D/16]
        Stage 4: [B, embed_dim*8,  H/32, W/32, D/32]
    """

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 48,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        window_size: tuple[int, ...] = (7, 7, 7),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()

        self.swin = SwinTransformer(
            in_chans=in_channels,
            embed_dim=embed_dim,
            window_size=window_size,
            patch_size=(4, 4, 4),
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=nn.LayerNorm,
            spatial_dims=3,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract hierarchical features.

        Args:
            x: Input volume [B, C, H, W, D]

        Returns:
            List of feature maps at each encoder stage (4 stages).
        """
        features = self.swin(x)
        # MONAI's SwinTransformer returns 5 features (4 stages + bottleneck).
        # We use only the 4 stage outputs matching our encoder_dims.
        return features[:4]
