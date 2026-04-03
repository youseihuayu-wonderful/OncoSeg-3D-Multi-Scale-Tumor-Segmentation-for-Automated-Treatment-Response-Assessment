"""Unit tests for model sub-modules."""

import torch


class TestCrossAttentionSkip:
    """Test cross-attention skip connection module."""

    def test_output_shape(self):
        from src.models.modules.cross_attention_skip import CrossAttentionSkip

        module = CrossAttentionSkip(encoder_dim=96, decoder_dim=96, num_heads=6)
        enc = torch.randn(1, 96, 8, 8, 8)
        dec = torch.randn(1, 96, 8, 8, 8)
        out = module(encoder_feat=enc, decoder_feat=dec)
        assert out.shape == (1, 96, 8, 8, 8)

    def test_different_spatial_same_channels(self):
        """Encoder and decoder can have same channel dim but cross-attention operates on them."""
        from src.models.modules.cross_attention_skip import CrossAttentionSkip

        module = CrossAttentionSkip(encoder_dim=48, decoder_dim=48, num_heads=3)
        enc = torch.randn(1, 48, 4, 4, 4)
        dec = torch.randn(1, 48, 4, 4, 4)
        out = module(encoder_feat=enc, decoder_feat=dec)
        assert out.shape == dec.shape

    def test_gradient_flows(self):
        from src.models.modules.cross_attention_skip import CrossAttentionSkip

        module = CrossAttentionSkip(encoder_dim=48, decoder_dim=48, num_heads=3)
        enc = torch.randn(1, 48, 4, 4, 4, requires_grad=True)
        dec = torch.randn(1, 48, 4, 4, 4, requires_grad=True)
        out = module(encoder_feat=enc, decoder_feat=dec)
        out.sum().backward()
        assert enc.grad is not None
        assert dec.grad is not None


class TestDeepSupervisionHead:
    """Test deep supervision auxiliary heads."""

    def test_output_count_and_shape(self):
        from src.models.modules.deep_supervision import DeepSupervisionHead

        head = DeepSupervisionHead(encoder_dims=[192, 96, 48], num_classes=4)
        intermediates = [
            torch.randn(1, 192, 4, 4, 4),
            torch.randn(1, 96, 8, 8, 8),
            torch.randn(1, 48, 16, 16, 16),
        ]
        outputs = head(intermediates)
        assert len(outputs) == 3
        # All outputs should be upsampled to finest resolution
        for out in outputs:
            assert out.shape[1] == 4  # num_classes
            assert out.shape[2:] == (16, 16, 16)  # finest resolution


class TestSwinEncoder:
    """Test Swin Transformer encoder."""

    def test_output_count(self):
        from src.models.modules.swin_encoder import SwinEncoder3D

        enc = SwinEncoder3D(
            in_channels=4,
            embed_dim=48,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
        )
        x = torch.randn(1, 4, 64, 64, 64)
        enc.eval()
        with torch.no_grad():
            features = enc(x)

        # Should return exactly 4 features (truncated from MONAI's 5)
        assert len(features) == 4
        assert features[0].shape[1] == 48  # embed_dim
        assert features[1].shape[1] == 96  # embed_dim * 2
        assert features[2].shape[1] == 192  # embed_dim * 4
        assert features[3].shape[1] == 384  # embed_dim * 8


class TestUNETRBaseline:
    """Test UNETR baseline model."""

    def test_forward_shape(self):
        from src.models.baselines.unetr import UNETR

        model = UNETR(in_channels=4, num_classes=4, img_size=(64, 64, 64))
        x = torch.randn(1, 4, 64, 64, 64)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out["pred"].shape == (1, 4, 64, 64, 64)
