"""Unit tests for RECIST measurement and response classification."""

import numpy as np
import pytest

from src.response.classifier import ResponseClassifier, ResponseCategory
from src.response.recist import RECISTMeasurer


class TestRECISTMeasurer:
    """Test RECIST 1.1 measurement module."""

    @pytest.fixture
    def measurer(self):
        return RECISTMeasurer()

    def test_empty_mask_diameter(self, measurer):
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        assert measurer.longest_axial_diameter(mask) == 0.0

    def test_empty_mask_volume(self, measurer):
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        assert measurer.volume_mm3(mask) == 0.0

    def test_single_voxel(self, measurer):
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[16, 16, 16] = 1
        assert measurer.volume_mm3(mask) == 1.0
        assert measurer.longest_axial_diameter(mask) == 0.0  # Single point has no diameter

    def test_cube_volume(self, measurer):
        """A 10×10×10 cube at 1mm spacing = 1000mm³."""
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[10:20, 10:20, 10:20] = 1
        vol = measurer.volume_mm3(mask, pixdim=(1.0, 1.0, 1.0))
        assert vol == 1000.0

    def test_anisotropic_spacing(self, measurer):
        """Volume should scale with voxel spacing."""
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[10:20, 10:20, 10:20] = 1  # 1000 voxels

        vol_iso = measurer.volume_mm3(mask, pixdim=(1.0, 1.0, 1.0))
        vol_aniso = measurer.volume_mm3(mask, pixdim=(2.0, 2.0, 2.0))

        assert vol_iso == 1000.0
        assert vol_aniso == 8000.0  # 2³ = 8× larger

    def test_measure_two_lesions(self, measurer):
        """Two separated lesions should be detected as 2 components."""
        mask = np.zeros((64, 64, 64), dtype=np.uint8)
        mask[10:20, 10:20, 10:20] = 1  # Lesion A
        mask[40:50, 40:50, 40:50] = 1  # Lesion B

        lesions = measurer.measure_lesions(mask)
        assert len(lesions) == 2
        # Both have same size → same volume
        assert lesions[0]["volume_mm3"] == lesions[1]["volume_mm3"] == 1000.0

    def test_no_lesions(self, measurer):
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        lesions = measurer.measure_lesions(mask)
        assert len(lesions) == 0


class TestResponseClassifier:
    """Test RECIST 1.1 response classification."""

    @pytest.fixture
    def classifier(self):
        return ResponseClassifier()

    def test_complete_response(self, classifier):
        """Tumor disappears completely → CR."""
        baseline = np.zeros((32, 32, 32), dtype=np.uint8)
        baseline[10:20, 10:20, 10:20] = 1
        followup = np.zeros((32, 32, 32), dtype=np.uint8)

        result = classifier.classify(baseline, followup)
        assert result.category == ResponseCategory.CR
        assert result.num_followup_lesions == 0

    def test_progressive_disease_growth(self, classifier):
        """Tumor grows ≥20% → PD."""
        baseline = np.zeros((64, 64, 64), dtype=np.uint8)
        baseline[20:30, 20:30, 20:30] = 1  # 10×10×10

        followup = np.zeros((64, 64, 64), dtype=np.uint8)
        followup[15:35, 15:35, 15:35] = 1  # 20×20×20 (much larger)

        result = classifier.classify(baseline, followup)
        assert result.category == ResponseCategory.PD
        assert result.percent_change > 0.20

    def test_progressive_disease_new_lesion(self, classifier):
        """New lesion appears → PD regardless of size change."""
        baseline = np.zeros((64, 64, 64), dtype=np.uint8)
        baseline[10:20, 10:20, 10:20] = 1  # 1 lesion

        followup = np.zeros((64, 64, 64), dtype=np.uint8)
        followup[10:20, 10:20, 10:20] = 1  # Same lesion
        followup[40:50, 40:50, 40:50] = 1  # New lesion

        result = classifier.classify(baseline, followup)
        assert result.category == ResponseCategory.PD
        assert result.new_lesions is True

    def test_stable_disease(self, classifier):
        """Same tumor size → SD."""
        baseline = np.zeros((64, 64, 64), dtype=np.uint8)
        baseline[20:30, 20:30, 20:30] = 1

        followup = np.zeros((64, 64, 64), dtype=np.uint8)
        followup[20:30, 20:30, 20:30] = 1  # Identical

        result = classifier.classify(baseline, followup)
        assert result.category == ResponseCategory.SD

    def test_no_baseline_tumor(self, classifier):
        """Both empty → technically CR (no disease)."""
        baseline = np.zeros((32, 32, 32), dtype=np.uint8)
        followup = np.zeros((32, 32, 32), dtype=np.uint8)

        result = classifier.classify(baseline, followup)
        assert result.category == ResponseCategory.CR
