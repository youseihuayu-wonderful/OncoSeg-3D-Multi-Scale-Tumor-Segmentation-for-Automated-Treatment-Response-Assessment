"""Temporal Attention module for longitudinal scan comparison."""

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Multi-head attention over temporal scan pairs for response assessment.

    Given segmentation features from baseline (t0) and follow-up (t1) scans,
    computes temporal attention to capture tumor evolution patterns.
    """

    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.temporal_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, feat_t0: torch.Tensor, feat_t1: torch.Tensor) -> torch.Tensor:
        """Compute temporal attention between two timepoints.

        Args:
            feat_t0: Baseline scan features [B, N, C]
            feat_t1: Follow-up scan features [B, N, C]

        Returns:
            Temporal difference features [B, N, C]
        """
        # Cross-attention: t1 attends to t0
        attended, _ = self.attention(query=feat_t1, key=feat_t0, value=feat_t0)
        attended = self.norm(attended + feat_t1)

        # Concatenate and project
        combined = torch.cat([attended, feat_t1 - feat_t0], dim=-1)
        temporal_feat = self.temporal_proj(combined)

        return temporal_feat
