"""Unit tests for RECIST measurement and response classification."""

import numpy as np
import pytest

from src.response.classifier import ResponseCategory, ResponseClassifier
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

    # --- RECIST 1.1 §3.1.1: 10mm minimum target lesion threshold ---

    def test_sub_threshold_lesion_excluded(self, measurer):
        """A 9mm lesion is below the RECIST 1.1 target threshold and must be dropped."""
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        # Line of 10 voxels at 1mm spacing -> longest diameter = 9mm
        mask[16, 10:20, 16] = 1
        lesions = measurer.measure_lesions(mask, pixdim=(1.0, 1.0, 1.0))
        assert len(lesions) == 0

    def test_threshold_lesion_kept(self, measurer):
        """A lesion with longest diameter exactly 10mm is kept (>= threshold)."""
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        # Line of 11 voxels at 1mm spacing -> diameter = 10mm exactly
        mask[16, 10:21, 16] = 1
        lesions = measurer.measure_lesions(mask, pixdim=(1.0, 1.0, 1.0))
        assert len(lesions) == 1
        assert lesions[0]["longest_diameter_mm"] == pytest.approx(10.0)

    def test_above_threshold_lesion_kept(self, measurer):
        """A 15mm lesion is kept."""
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        # Line of 16 voxels at 1mm spacing -> diameter = 15mm
        mask[16, 8:24, 16] = 1
        lesions = measurer.measure_lesions(mask, pixdim=(1.0, 1.0, 1.0))
        assert len(lesions) == 1
        assert lesions[0]["longest_diameter_mm"] == pytest.approx(15.0)

    def test_mixed_threshold_filtering(self, measurer):
        """Two lesions: one 15mm (kept), one 9mm (dropped) -> only 1 returned."""
        mask = np.zeros((64, 64, 64), dtype=np.uint8)
        mask[10, 5:21, 10] = 1  # 16 voxels -> 15mm (above threshold)
        mask[40, 40:50, 40] = 1  # 10 voxels -> 9mm (below threshold)
        lesions = measurer.measure_lesions(mask, pixdim=(1.0, 1.0, 1.0))
        assert len(lesions) == 1
        assert lesions[0]["longest_diameter_mm"] == pytest.approx(15.0)

    # --- RECIST 1.1 §3.1.2: max 5 target lesions ---

    def test_max_target_lesions_cap(self, measurer):
        """7 valid (>=10mm) lesions in input -> exactly 5 returned (the 5 largest)."""
        mask = np.zeros((128, 128, 128), dtype=np.uint8)
        # 7 disconnected line lesions with distinct longest diameters
        # voxel counts: 11, 12, 13, 14, 15, 16, 17 -> diameters 10, 11, 12, 13, 14, 15, 16 mm
        for k, n_vox in enumerate([11, 12, 13, 14, 15, 16, 17]):
            row = 5 + 15 * k
            mask[row, 5 : 5 + n_vox, 5] = 1

        lesions = measurer.measure_lesions(mask, pixdim=(1.0, 1.0, 1.0))
        assert len(lesions) == RECISTMeasurer.MAX_TARGET_LESIONS == 5
        diameters = [les["longest_diameter_mm"] for les in lesions]
        # Should be the five largest: 12, 13, 14, 15, 16 mm, sorted descending
        assert diameters == pytest.approx([16.0, 15.0, 14.0, 13.0, 12.0])


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


def _stub_measure_lesions_factory(mask_to_lesions: dict):
    """Build a stand-in for RECISTMeasurer.measure_lesions keyed by the mask object id.

    Each entry maps id(mask) -> list of lesion dicts to return. This lets us drive
    the classifier with exact, hand-chosen sums of longest diameters.
    """

    def _stub(self, mask, pixdim=(1.0, 1.0, 1.0)):  # noqa: ARG001
        return mask_to_lesions[id(mask)]

    return _stub


def _make_lesion(diameter_mm: float, lesion_id: int = 1) -> dict:
    """Build a lesion dict with a given longest diameter (volume kept consistent)."""
    return {
        "id": lesion_id,
        "longest_diameter_mm": float(diameter_mm),
        "volume_mm3": float(diameter_mm) ** 3,  # arbitrary; unused for category logic
        "voxel_count": int(diameter_mm),
    }


class TestProgressiveDiseaseAbsoluteRule:
    """RECIST 1.1 §4.3: PD requires BOTH >=20% relative AND >=5mm absolute increase."""

    @pytest.fixture
    def classifier(self):
        return ResponseClassifier()

    def test_relative_threshold_met_but_absolute_not_is_sd(self, classifier, monkeypatch):
        """+21% but only +4mm absolute -> SD, not PD."""
        baseline = np.zeros((4, 4, 4), dtype=np.uint8)
        followup = np.zeros((4, 4, 4), dtype=np.uint8)
        # baseline sum = 19.0 mm, followup = 23.0 mm -> +4.0 mm, +21.05%
        lesion_map = {
            id(baseline): [_make_lesion(19.0)],
            id(followup): [_make_lesion(23.0)],
        }
        monkeypatch.setattr(
            "src.response.recist.RECISTMeasurer.measure_lesions",
            _stub_measure_lesions_factory(lesion_map),
        )
        result = classifier.classify(baseline, followup)
        assert result.percent_change > 0.20
        assert (result.followup_sum_ld - result.baseline_sum_ld) < 5.0
        assert result.category == ResponseCategory.SD

    def test_relative_and_absolute_thresholds_met_is_pd(self, classifier, monkeypatch):
        """+21% and +6mm absolute -> PD."""
        baseline = np.zeros((4, 4, 4), dtype=np.uint8)
        followup = np.zeros((4, 4, 4), dtype=np.uint8)
        # baseline 28.0 mm, followup 34.0 mm -> +6.0 mm, +21.4%
        lesion_map = {
            id(baseline): [_make_lesion(28.0)],
            id(followup): [_make_lesion(34.0)],
        }
        monkeypatch.setattr(
            "src.response.recist.RECISTMeasurer.measure_lesions",
            _stub_measure_lesions_factory(lesion_map),
        )
        result = classifier.classify(baseline, followup)
        assert result.percent_change > 0.20
        assert (result.followup_sum_ld - result.baseline_sum_ld) >= 5.0
        assert result.category == ResponseCategory.PD

    def test_pd_boundary_exact_20pct_and_exact_5mm(self, classifier, monkeypatch):
        """Exactly +20% and exactly +5mm -> PD (boundary case)."""
        baseline = np.zeros((4, 4, 4), dtype=np.uint8)
        followup = np.zeros((4, 4, 4), dtype=np.uint8)
        # baseline 25.0 mm, followup 30.0 mm -> +5.0 mm, +20.00%
        lesion_map = {
            id(baseline): [_make_lesion(25.0)],
            id(followup): [_make_lesion(30.0)],
        }
        monkeypatch.setattr(
            "src.response.recist.RECISTMeasurer.measure_lesions",
            _stub_measure_lesions_factory(lesion_map),
        )
        result = classifier.classify(baseline, followup)
        assert result.percent_change == pytest.approx(0.20)
        assert (result.followup_sum_ld - result.baseline_sum_ld) == pytest.approx(5.0)
        assert result.category == ResponseCategory.PD


class TestRECISTNadir:
    """RECIST 1.1 §4.3: PD is computed against the nadir, not the baseline."""

    @pytest.fixture
    def classifier(self):
        return ResponseClassifier()

    def test_nadir_pd_classic_failure(self, classifier, monkeypatch):
        """baseline=100 -> t1=70 (PR) -> t2=85.

        Without history, t2 vs baseline is -15% -> SD.
        With history=[70.0], t2 vs nadir(70) is +21.4% and +15mm -> PD.
        """
        baseline = np.zeros((4, 4, 4), dtype=np.uint8)
        followup_t2 = np.zeros((4, 4, 4), dtype=np.uint8)
        lesion_map = {
            id(baseline): [_make_lesion(100.0)],
            id(followup_t2): [_make_lesion(85.0)],
        }
        monkeypatch.setattr(
            "src.response.recist.RECISTMeasurer.measure_lesions",
            _stub_measure_lesions_factory(lesion_map),
        )

        # Without nadir history -> SD
        result_no_history = classifier.classify(baseline, followup_t2)
        assert result_no_history.category == ResponseCategory.SD

        # With empty history -> still SD (defensive: empty list == None semantics)
        result_empty_history = classifier.classify(baseline, followup_t2, history_sums=[])
        assert result_empty_history.category == ResponseCategory.SD

        # With history including the t1 nadir of 70 mm -> PD
        result_with_nadir = classifier.classify(baseline, followup_t2, history_sums=[70.0])
        assert result_with_nadir.category == ResponseCategory.PD

    def test_nadir_does_not_affect_pr_classification(self, classifier, monkeypatch):
        """PR/SD/CR continue to compare against baseline, not nadir."""
        baseline = np.zeros((4, 4, 4), dtype=np.uint8)
        followup = np.zeros((4, 4, 4), dtype=np.uint8)
        # baseline 100 mm, followup 60 mm -> -40% vs baseline -> PR.
        # If nadir-history of [50] were (incorrectly) used for PR, current would be
        # +20% vs nadir, but PR must still resolve via baseline comparison.
        lesion_map = {
            id(baseline): [_make_lesion(100.0)],
            id(followup): [_make_lesion(60.0)],
        }
        monkeypatch.setattr(
            "src.response.recist.RECISTMeasurer.measure_lesions",
            _stub_measure_lesions_factory(lesion_map),
        )
        result = classifier.classify(baseline, followup, history_sums=[50.0])
        assert result.category == ResponseCategory.PR

    def test_nadir_baseline_used_when_baseline_is_smallest(self, classifier, monkeypatch):
        """If baseline is the smallest SoD seen, baseline is the nadir."""
        baseline = np.zeros((4, 4, 4), dtype=np.uint8)
        followup = np.zeros((4, 4, 4), dtype=np.uint8)
        # baseline 50, t1=80 (history), t2 (followup) = 60 -> vs nadir(50) +20% +10mm -> PD
        lesion_map = {
            id(baseline): [_make_lesion(50.0)],
            id(followup): [_make_lesion(60.0)],
        }
        monkeypatch.setattr(
            "src.response.recist.RECISTMeasurer.measure_lesions",
            _stub_measure_lesions_factory(lesion_map),
        )
        result = classifier.classify(baseline, followup, history_sums=[80.0])
        assert result.category == ResponseCategory.PD
