"""CNN Decoder with transposed convolution upsampling for 3D segmentation."""

import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    """Single decoder block: upsample + cross-attention skip + conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()

        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        cross_attn: nn.Module | None = None,
    ) -> torch.Tensor:
        """Upsample, fuse with skip connection, and refine.

        Args:
            x: Input features from deeper decoder stage [B, C_in, H, W, D]
            skip: Encoder skip features [B, C_skip, 2H, 2W, 2D]
            cross_attn: Optional cross-attention module for skip fusion

        Returns:
            Decoded features [B, C_out, 2H, 2W, 2D]
        """
        x = self.upsample(x)

        if cross_attn is not None:
            x = cross_attn(encoder_feat=skip, decoder_feat=x)
        else:
            x = x + skip

        x = self.conv(x)
        return x


class CNNDecoder3D(nn.Module):
    """Full CNN decoder path with multi-scale upsampling."""

    def __init__(self, encoder_dims: list[int], num_classes: int):
        super().__init__()

        self.blocks = nn.ModuleList()
        reversed_dims = list(reversed(encoder_dims))

        for i in range(len(reversed_dims) - 1):
            self.blocks.append(
                DecoderBlock(
                    in_channels=reversed_dims[i],
                    skip_channels=reversed_dims[i + 1],
                    out_channels=reversed_dims[i + 1],
                )
            )

        # Recover the 4x downsampling from patch embedding (2x + 2x = 4x)
        self.upsample_head = nn.Sequential(
            nn.ConvTranspose3d(encoder_dims[0], encoder_dims[0], kernel_size=2, stride=2),
            nn.InstanceNorm3d(encoder_dims[0]),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(encoder_dims[0], encoder_dims[0], kernel_size=2, stride=2),
            nn.InstanceNorm3d(encoder_dims[0]),
            nn.LeakyReLU(inplace=True),
        )

        self.final_conv = nn.Conv3d(encoder_dims[0], num_classes, kernel_size=1)

    def forward(
        self,
        encoder_features: list[torch.Tensor],
        cross_attn_skips: nn.ModuleList,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Decode encoder features into segmentation map.

        Args:
            encoder_features: List of encoder stage outputs [stage1, ..., stage4]
            cross_attn_skips: Cross-attention modules for each skip connection

        Returns:
            Dict with "pred" (final output) and "intermediate" (for deep supervision)
        """
        x = encoder_features[-1]  # Bottleneck
        intermediates = []

        for i, block in enumerate(self.blocks):
            skip_idx = len(encoder_features) - 2 - i
            skip = encoder_features[skip_idx]
            cross_attn = cross_attn_skips[skip_idx] if cross_attn_skips else None
            x = block(x, skip, cross_attn)
            intermediates.append(x)

        # Upsample to recover the 4x patch embedding downsampling
        x = self.upsample_head(x)

        pred = self.final_conv(x)

        return {"pred": pred, "intermediate": intermediates}
