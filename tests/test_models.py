"""Unit tests for model architectures."""

import pytest
import torch


class TestOncoSeg:
    """Test OncoSeg model forward pass and output shapes."""

    @pytest.fixture
    def model(self):
        from src.models.oncoseg import OncoSeg
        return OncoSeg(
            in_channels=4,
            num_classes=4,
            embed_dim=48,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            deep_supervision=True,
        )

    def test_output_shape(self, model):
        """Verify output spatial dimensions match input."""
        x = torch.randn(1, 4, 128, 128, 128)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out["pred"].shape == (1, 4, 128, 128, 128)

    def test_deep_supervision_training(self, model):
        """Verify deep supervision outputs during training."""
        x = torch.randn(1, 4, 128, 128, 128)
        model.train()
        out = model(x)
        assert "pred" in out
        assert "deep_sup" in out
        assert len(out["deep_sup"]) > 0


class TestBaselines:
    """Test baseline model forward passes."""

    def test_unet3d(self):
        from src.models.baselines.unet3d import UNet3D
        model = UNet3D(in_channels=4, num_classes=4)
        x = torch.randn(1, 4, 128, 128, 128)
        out = model(x)
        assert out["pred"].shape == (1, 4, 128, 128, 128)

    def test_swin_unetr(self):
        from src.models.baselines.swin_unetr import SwinUNETRBaseline
        model = SwinUNETRBaseline(in_channels=4, num_classes=4, img_size=(128, 128, 128))
        x = torch.randn(1, 4, 128, 128, 128)
        out = model(x)
        assert out["pred"].shape == (1, 4, 128, 128, 128)


class TestRECIST:
    """Test RECIST measurement module."""

    def test_empty_mask(self):
        import numpy as np
        from src.response.recist import RECISTMeasurer
        measurer = RECISTMeasurer()
        mask = np.zeros((64, 64, 64), dtype=np.uint8)
        assert measurer.longest_axial_diameter(mask) == 0.0
        assert measurer.volume_mm3(mask) == 0.0

    def test_sphere_volume(self):
        import numpy as np
        from src.response.recist import RECISTMeasurer
        measurer = RECISTMeasurer()

        # Create a sphere with radius 10 voxels
        mask = np.zeros((64, 64, 64), dtype=np.uint8)
        center = np.array([32, 32, 32])
        for x in range(64):
            for y in range(64):
                for z in range(64):
                    if np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= 10:
                        mask[x, y, z] = 1

        vol = measurer.volume_mm3(mask, pixdim=(1.0, 1.0, 1.0))
        expected = (4 / 3) * np.pi * 10**3
        # Allow 10% tolerance for discrete voxel approximation
        assert abs(vol - expected) / expected < 0.10

    def test_response_classifier(self):
        import numpy as np
        from src.response.classifier import ResponseClassifier, ResponseCategory

        classifier = ResponseClassifier()

        # Baseline: large tumor, Follow-up: no tumor → Complete Response
        baseline = np.zeros((64, 64, 64), dtype=np.uint8)
        baseline[20:40, 20:40, 20:40] = 1
        followup = np.zeros((64, 64, 64), dtype=np.uint8)

        result = classifier.classify(baseline, followup)
        assert result.category == ResponseCategory.CR
