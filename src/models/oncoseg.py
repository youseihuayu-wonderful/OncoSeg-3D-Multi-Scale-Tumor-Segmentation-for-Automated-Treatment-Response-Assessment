"""OncoSeg: Hybrid Swin Transformer-CNN U-Net for 3D Tumor Segmentation."""

import torch
import torch.nn as nn

from src.models.modules.cnn_decoder import CNNDecoder3D
from src.models.modules.cross_attention_skip import CrossAttentionSkip
from src.models.modules.deep_supervision import DeepSupervisionHead
from src.models.modules.swin_encoder import SwinEncoder3D
from src.models.modules.temporal_attention import TemporalAttention


class OncoSeg(nn.Module):
    """Hybrid Swin Transformer encoder + CNN decoder with cross-attention skip connections.

    Architecture:
        - Encoder: 3D Swin Transformer (4 stages, shifted window self-attention)
        - Skip Connections: Cross-attention fusion (decoder queries encoder features)
        - Decoder: CNN upsampling path (transposed conv3d)
        - Head: Multi-class segmentation + deep supervision
        - Uncertainty: MC Dropout for confidence estimation
        - Temporal: Optional temporal attention for longitudinal scan comparison
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        embed_dim: int = 48,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        window_size: tuple[int, ...] = (7, 7, 7),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_rate: float = 0.1,
        deep_supervision: bool = True,
        temporal: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.temporal = temporal

        # Encoder: 3D Swin Transformer
        self.encoder = SwinEncoder3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )

        # Feature dimensions at each encoder stage
        encoder_dims = [embed_dim * (2**i) for i in range(len(depths))]

        # Cross-attention skip connections
        self.cross_attn_skips = nn.ModuleList([
            CrossAttentionSkip(
                encoder_dim=encoder_dims[i],
                decoder_dim=encoder_dims[i],
                num_heads=num_heads[i],
            )
            for i in range(len(depths) - 1)
        ])

        # Decoder: CNN upsampling path
        self.decoder = CNNDecoder3D(
            encoder_dims=encoder_dims,
            num_classes=num_classes,
        )

        # Deep supervision heads
        if deep_supervision:
            # Decoder intermediates are in reverse order (deep→shallow):
            # [embed_dim*4, embed_dim*2, embed_dim]
            self.ds_heads = DeepSupervisionHead(
                encoder_dims=list(reversed(encoder_dims[:-1])),
                num_classes=num_classes,
            )

        # MC Dropout for uncertainty estimation
        self.mc_dropout = nn.Dropout3d(p=dropout_rate)

        # Temporal attention for longitudinal comparison (baseline vs follow-up)
        if temporal:
            bottleneck_dim = encoder_dims[-1]
            self.temporal_attn = TemporalAttention(
                feature_dim=bottleneck_dim,
                num_heads=num_heads[-1],
            )

    def forward(
        self,
        x: torch.Tensor,
        x_followup: torch.Tensor | None = None,
        mc_samples: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input volume [B, C, H, W, D] (baseline scan)
            x_followup: Optional follow-up volume [B, C, H, W, D] for temporal mode.
                        When provided, temporal attention fuses bottleneck features
                        from both timepoints before decoding.
            mc_samples: Number of MC Dropout forward passes for uncertainty.
                        0 = deterministic inference, >0 = uncertainty estimation.

        Returns:
            Dictionary with keys:
                - "pred": Main segmentation prediction [B, num_classes, H, W, D]
                - "deep_sup": List of deep supervision outputs (if training + deep_supervision)
                - "uncertainty": Uncertainty map [B, 1, H, W, D] (if mc_samples > 0)
                - "temporal_feat": Temporal difference features (if x_followup provided)
        """
        # Encode baseline
        encoder_features = self.encoder(x)

        # Temporal attention: fuse baseline and follow-up bottleneck features
        if self.temporal and x_followup is not None:
            followup_features = self.encoder(x_followup)
            encoder_features, temporal_feat = self._temporal_fusion(
                encoder_features, followup_features
            )

        # Decode with cross-attention skip connections
        decoder_features = self.decoder(encoder_features, self.cross_attn_skips)

        outputs = {"pred": decoder_features["pred"]}

        # Deep supervision
        if self.deep_supervision and self.training:
            outputs["deep_sup"] = self.ds_heads(decoder_features["intermediate"])

        # Temporal features for downstream response analysis
        if self.temporal and x_followup is not None:
            outputs["temporal_feat"] = temporal_feat

        # MC Dropout uncertainty estimation
        if mc_samples > 0:
            outputs["uncertainty"] = self._mc_uncertainty(x, mc_samples)

        return outputs

    def _temporal_fusion(
        self,
        baseline_features: list[torch.Tensor],
        followup_features: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Fuse baseline and follow-up features via temporal attention.

        Operates on the bottleneck (deepest) features. The temporal attention
        module computes cross-attention from follow-up to baseline, capturing
        what changed between timepoints.

        Returns:
            - Modified encoder features (bottleneck replaced with temporal-fused)
            - Raw temporal difference features for downstream analysis
        """
        bottleneck_t0 = baseline_features[-1]   # [B, C, H, W, D]
        bottleneck_t1 = followup_features[-1]

        B, C, H, W, D = bottleneck_t0.shape

        # Reshape to sequence for temporal attention: [B, N, C]
        feat_t0 = bottleneck_t0.flatten(2).transpose(1, 2)
        feat_t1 = bottleneck_t1.flatten(2).transpose(1, 2)

        # Temporal attention
        temporal_feat = self.temporal_attn(feat_t0, feat_t1)

        # Reshape back to volume
        fused_bottleneck = temporal_feat.transpose(1, 2).reshape(B, C, H, W, D)

        # Replace bottleneck with temporal-fused features
        fused_features = list(baseline_features)
        fused_features[-1] = fused_bottleneck

        return fused_features, fused_bottleneck

    def _mc_uncertainty(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Estimate prediction uncertainty via MC Dropout."""
        self.mc_dropout.train()  # Keep dropout active
        predictions = []

        for _ in range(n_samples):
            enc_features = self.encoder(x)
            # Apply dropout to bottleneck
            enc_features[-1] = self.mc_dropout(enc_features[-1])
            dec_out = self.decoder(enc_features, self.cross_attn_skips)
            predictions.append(torch.softmax(dec_out["pred"], dim=1))

        stacked = torch.stack(predictions, dim=0)  # [N, B, C, H, W, D]
        mean_pred = stacked.mean(dim=0)
        uncertainty = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=1, keepdim=True)

        return uncertainty
