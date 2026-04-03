"""Cross-Attention Skip Connection module.

Replaces standard concatenation-based skip connections with cross-attention,
allowing the decoder to selectively query relevant encoder features.
"""

import torch
import torch.nn as nn


class CrossAttentionSkip(nn.Module):
    """Cross-attention skip connection between encoder and decoder features.

    The decoder features serve as queries, and encoder features serve as
    keys and values. This allows the decoder to attend to the most relevant
    spatial locations in the encoder output.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = decoder_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(decoder_dim, decoder_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(encoder_dim, decoder_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(encoder_dim, decoder_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(decoder_dim, decoder_dim)

        self.norm_enc = nn.LayerNorm(encoder_dim)
        self.norm_dec = nn.LayerNorm(decoder_dim)
        self.norm_out = nn.LayerNorm(decoder_dim)

        self.ffn = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim * 4),
            nn.GELU(),
            nn.Linear(decoder_dim * 4, decoder_dim),
        )

    def forward(self, encoder_feat: torch.Tensor, decoder_feat: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention between encoder and decoder features.

        Args:
            encoder_feat: Encoder features [B, C_enc, H, W, D]
            decoder_feat: Decoder features [B, C_dec, H, W, D]

        Returns:
            Fused features [B, C_dec, H, W, D]
        """
        B, C_dec, H, W, D = decoder_feat.shape

        # Reshape to sequence: [B, H*W*D, C]
        enc_seq = encoder_feat.flatten(2).transpose(1, 2)  # [B, N, C_enc]
        dec_seq = decoder_feat.flatten(2).transpose(1, 2)  # [B, N, C_dec]

        # Normalize
        enc_seq = self.norm_enc(enc_seq)
        dec_seq = self.norm_dec(dec_seq)

        # Cross-attention: decoder queries encoder
        Q = self.q_proj(dec_seq).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(enc_seq).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(enc_seq).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ V).transpose(1, 2).reshape(B, -1, C_dec)
        out = self.out_proj(out)

        # Residual + FFN
        out = dec_seq + out
        out = out + self.ffn(self.norm_out(out))

        # Reshape back to volume
        out = out.transpose(1, 2).reshape(B, C_dec, H, W, D)

        return out
