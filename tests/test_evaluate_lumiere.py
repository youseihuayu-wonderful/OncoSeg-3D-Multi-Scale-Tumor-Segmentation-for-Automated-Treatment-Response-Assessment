"""Tests for scripts/evaluate_lumiere.

These tests exercise the logic the script layers on top of the real
inference / response-classification pipeline — RANO normalisation, RECIST
→ RANO agreement aggregation, Cohen kappa, confusion-matrix construction,
per-visit record emission, padding of mis-shaped masks, and end-to-end
orchestration with a deterministic in-process predictor.

We do not mock the ``Predictor`` class. The tests build a real
``Predictor`` around a tiny ``nn.Module`` whose forward pass returns
pre-computed logits that decode into known ET masks. That keeps the test
fast and self-contained while still running the real sliding-window
inference, sigmoid thresholding, RECIST measurement, and response
classification paths.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_lumiere as el  # noqa: E402
from src.response.classifier import ResponseCategory  # noqa: E402


# ---------------------------------------------------------------------------
# Pure-Python helpers
# ---------------------------------------------------------------------------


class TestRanoMapping:
    def test_baseline_is_not_a_response(self):
        assert el.rano_to_recist("baseline") is None
        assert el.rano_to_recist("Baseline") is None

    def test_pre_rt_variants_treated_as_non_response(self):
        for label in ("Pre-RT", "pre_rt", "PreRT"):
            assert el.rano_to_recist(label) is None

    def test_empty_and_none_are_none(self):
        assert el.rano_to_recist(None) is None
        assert el.rano_to_recist("") is None
        assert el.rano_to_recist("   ") is None

    def test_canonical_abbreviations_map_to_recist(self):
        assert el.rano_to_recist("CR") is ResponseCategory.CR
        assert el.rano_to_recist("pr") is ResponseCategory.PR
        assert el.rano_to_recist("SD") is ResponseCategory.SD
        assert el.rano_to_recist("PD") is ResponseCategory.PD

    def test_full_spellings_map_to_recist(self):
        assert el.rano_to_recist("Partial Response") is ResponseCategory.PR
        assert el.rano_to_recist("Progression") is ResponseCategory.PD

    def test_unknown_label_returns_none(self):
        assert el.rano_to_recist("mixed response") is None


class TestCohenKappa:
    def test_perfect_agreement_is_one(self):
        assert el.cohen_kappa(["CR", "PR", "SD", "PD"], ["CR", "PR", "SD", "PD"]) == 1.0

    def test_total_disagreement_on_balanced_labels_is_nonpositive(self):
        kappa = el.cohen_kappa(["CR", "CR", "PD", "PD"], ["PD", "PD", "CR", "CR"])
        assert kappa <= 0.0

    def test_empty_input_returns_nan(self):
        assert np.isnan(el.cohen_kappa([], []))

    def test_chance_agreement_on_single_label_is_defined(self):
        # When both judges always say the same single label, by-chance agreement
        # is already 1.0; kappa is conventionally 1.0 if observed == expected == 1.
        assert el.cohen_kappa(["CR", "CR", "CR"], ["CR", "CR", "CR"]) == 1.0


class TestConfusionMatrix:
    def test_diagonal_is_correct_count(self):
        mat = el.confusion_matrix(["CR", "PR", "SD", "PD"], ["CR", "PR", "SD", "PD"],
                                  labels=["CR", "PR", "SD", "PD"])
        assert np.array_equal(np.diag(mat), [1, 1, 1, 1])

    def test_off_diagonal_records_disagreement(self):
        mat = el.confusion_matrix(["CR"], ["PD"], labels=["CR", "PR", "SD", "PD"])
        assert mat[0, 3] == 1

    def test_ignores_unknown_labels(self):
        mat = el.confusion_matrix(["UNKNOWN"], ["CR"], labels=["CR", "PR", "SD", "PD"])
        assert mat.sum() == 0


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


class TestPadToCommon:
    def test_pads_smaller_array_up(self):
        a = np.ones((4, 4, 4), dtype=np.uint8)
        b = np.ones((2, 4, 4), dtype=np.uint8)
        pa, pb = el._pad_to_common(a, b)
        assert pa.shape == pb.shape == (4, 4, 4)
        assert pa.sum() == 4 * 4 * 4
        assert pb.sum() == 2 * 4 * 4  # padding adds zeros

    def test_identical_shapes_unchanged(self):
        a = np.zeros((3, 3, 3), dtype=np.uint8)
        b = np.ones((3, 3, 3), dtype=np.uint8)
        pa, pb = el._pad_to_common(a, b)
        assert pa.shape == pb.shape == (3, 3, 3)
        assert pb.sum() == 27


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


def _record(patient, fu, predicted, expert, agree):
    return el.VisitRecord(
        patient_id=patient,
        baseline_tp="week-000",
        followup_tp=fu,
        baseline_sum_ld_mm=50.0,
        followup_sum_ld_mm=30.0,
        percent_change=-0.4,
        num_baseline_lesions=1,
        num_followup_lesions=1,
        predicted=predicted,
        expert_rano=expert,
        expert_rationale=None,
        agree=agree,
    )


class TestSummarise:
    def test_all_agree_yields_accuracy_one_and_kappa_one(self):
        records = [
            _record("Patient-01", "week-013", "CR", "CR", True),
            _record("Patient-01", "week-024", "PR", "PR", True),
            _record("Patient-02", "week-013", "SD", "SD", True),
            _record("Patient-02", "week-024", "PD", "PD", True),
        ]
        summary = el.summarise(records)
        assert summary["accuracy"] == 1.0
        assert summary["cohen_kappa"] == 1.0
        assert summary["total_visits"] == 4
        assert summary["comparable_visits"] == 4

    def test_missing_expert_rating_excluded_from_metrics(self):
        records = [
            _record("Patient-01", "week-013", "PR", "PR", True),
            _record("Patient-01", "week-024", "SD", None, None),  # no expert → excluded
        ]
        summary = el.summarise(records)
        assert summary["total_visits"] == 2
        assert summary["comparable_visits"] == 1
        assert summary["accuracy"] == 1.0

    def test_no_comparable_returns_none_metrics(self):
        records = [_record("Patient-01", "week-013", "SD", None, None)]
        summary = el.summarise(records)
        assert summary["accuracy"] is None
        assert summary["cohen_kappa"] is None
        assert "note" in summary

    def test_mixed_accuracy(self):
        records = [
            _record("Patient-01", "week-013", "PR", "PR", True),
            _record("Patient-01", "week-024", "SD", "PD", False),
            _record("Patient-02", "week-013", "SD", "SD", True),
            _record("Patient-02", "week-024", "CR", "PR", False),
        ]
        summary = el.summarise(records)
        assert summary["accuracy"] == pytest.approx(0.5)
        assert 0 <= summary["cohen_kappa"] <= 1


class TestWritePerVisitCsv:
    def test_emits_header_and_rows(self, tmp_path):
        records = [
            _record("Patient-01", "week-013", "PR", "PR", True),
            _record("Patient-02", "week-013", "SD", "SD", True),
        ]
        path = tmp_path / "per_visit.csv"
        el.write_per_visit_csv(records, path)
        lines = path.read_text().splitlines()
        assert lines[0].startswith("patient_id,baseline_tp,followup_tp")
        assert len(lines) == 3  # header + 2 rows
        assert "Patient-01" in lines[1]

    def test_empty_records_writes_empty_file(self, tmp_path):
        path = tmp_path / "per_visit.csv"
        el.write_per_visit_csv([], path)
        assert path.read_text() == ""


# ---------------------------------------------------------------------------
# End-to-end with a deterministic in-process predictor.
# ---------------------------------------------------------------------------


class _FixedETModel(nn.Module):
    """Tiny model whose forward() returns logits that decode to a given ET mask.

    The network is *real* (subclass of ``nn.Module``) and runs on CPU. It does
    not perform any learning — its output is determined by the spatial extent
    we prescribe via ``et_fraction``, which fixes what fraction of the central
    axial region is labelled enhancing tumor. This lets us script CR / PR /
    SD / PD scenarios without training weights while still flowing through
    the real sliding-window inference, sigmoid threshold, and RECIST logic.
    """

    def __init__(self, et_fraction: float):
        super().__init__()
        self.et_fraction = float(et_fraction)
        # A parameter so ``.to(device)`` works without surprises.
        self.register_buffer("_dummy", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, h, w, d = x.shape
        logits = torch.full((b, 3, h, w, d), -10.0, device=x.device, dtype=x.dtype)
        # Central cube whose side length is et_fraction * min_dim. Small ET → CR-ish;
        # large ET → PD-ish.
        if self.et_fraction > 0:
            side = max(1, int(round(self.et_fraction * min(h, w, d))))
            hs, ws, ds = (h - side) // 2, (w - side) // 2, (d - side) // 2
            logits[:, 2, hs:hs + side, ws:ws + side, ds:ds + side] = 10.0  # ET ON
        return {"pred": logits}


def _write_nii(path: Path, shape=(16, 16, 16), seed: int = 0):
    """Write a NIfTI with a gentle intensity gradient so NormalizeIntensityd
    (z-score over non-zero voxels) produces finite, varied values and
    CropForegroundd does not crop the entire volume away.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Gradient + small noise; all voxels are strictly positive so the whole
    # volume is treated as foreground.
    grid = np.indices(shape, dtype=np.float32).sum(axis=0) + 1.0
    data = grid + rng.normal(0.0, 0.1, size=shape).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def _make_lumiere_patient(root: Path, patient_id: str, weeks: list[int], rano_per_week: dict[int, str]):
    """Lay down a synthetic LUMIERE patient. Appends rows to the shared
    clinical CSV rather than overwriting it, so multiple patients can be
    added to the same root in sequence.
    """
    modality_filenames = {
        "t1n": "t1_skull_strip.nii.gz",
        "t1c": "ct1_skull_strip.nii.gz",
        "t2w": "t2_skull_strip.nii.gz",
        "t2f": "flair_skull_strip.nii.gz",
    }
    pdir = root / patient_id
    for w in weeks:
        strip = pdir / f"week-{w:03d}" / "DeepBraTumIA-segmentation/atlas/skull_strip"
        for i, (_mod, fname) in enumerate(modality_filenames.items()):
            _write_nii(strip / fname, seed=hash((patient_id, w, i)) & 0xFFFF)

    csv_path = root / "LumiereClinicalData.csv"
    if not csv_path.exists():
        csv_path.write_text("patient_id,week,rano,rationale\n")
    with csv_path.open("a", encoding="utf-8") as fh:
        for w, r in rano_per_week.items():
            fh.write(f"{patient_id},{w},{r},test\n")


class TestEndToEnd:
    def test_full_run_emits_artefacts_and_correct_agreement(self, tmp_path, monkeypatch):
        # Two patients: Patient-01 has a large baseline that shrinks (PR expected),
        # Patient-02 is stable (SD expected). We wire up the model to produce
        # exactly those shrink patterns regardless of input.
        lumiere_root = tmp_path / "LUMIERE"
        _make_lumiere_patient(
            lumiere_root,
            "Patient-01",
            weeks=[0, 13],
            rano_per_week={0: "baseline", 13: "PR"},
        )
        _make_lumiere_patient(
            lumiere_root,
            "Patient-02",
            weeks=[0, 13],
            rano_per_week={0: "baseline", 13: "SD"},
        )

        # Per-timepoint ET fractions chosen to trip the RECIST thresholds:
        #   PR requires >= 30% decrease in sum-LD.
        #   SD requires < 20% increase and no PR-level shrinkage.
        fractions = {
            ("Patient-01", "week-000"): 0.60,  # baseline large
            ("Patient-01", "week-013"): 0.30,  # half the side → shrinkage well past 30%
            ("Patient-02", "week-000"): 0.40,
            ("Patient-02", "week-013"): 0.42,  # barely larger → SD
        }

        # Patch load_oncoseg so no real checkpoint is needed; stand up a fresh
        # _FixedETModel whose et_fraction depends on the current timepoint.
        call_state = {"current_tp": None}

        class RouterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("_dummy", torch.zeros(1))

            def forward(self, x):
                key = call_state["current_tp"]
                frac = fractions.get(key, 0.0)
                return _FixedETModel(frac)(x)

        def fake_load(checkpoint_path, device):
            return RouterModel().to(device).eval()

        monkeypatch.setattr(el, "load_oncoseg", fake_load)

        # Hook segment_timepoint to record which (patient, timepoint) is being
        # processed so RouterModel can pick the right fraction.
        real_segment = el.segment_timepoint

        def traced_segment(predictor, transforms, tp):
            call_state["current_tp"] = (tp.patient_id, tp.timepoint_id)
            return real_segment(predictor, transforms, tp)

        monkeypatch.setattr(el, "segment_timepoint", traced_segment)

        output_dir = tmp_path / "out"
        summary = el.run(
            lumiere_root=lumiere_root,
            checkpoint=tmp_path / "dummy.pth",  # never read
            output_dir=output_dir,
            max_patients=None,
            roi_size=(16, 16, 16),
            pixdim=(1.0, 1.0, 1.0),
            mc_samples=0,
        )

        # Artefacts
        assert (output_dir / "per_visit.csv").exists()
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "confusion_matrix.png").exists()
        assert (output_dir / "run.log").exists()

        # Summary shape
        persisted = json.loads((output_dir / "summary.json").read_text())
        assert persisted == summary
        assert summary["total_visits"] == 2
        assert summary["comparable_visits"] == 2
        assert summary["accuracy"] == 1.0

        # Per-visit CSV sanity
        csv_text = (output_dir / "per_visit.csv").read_text().splitlines()
        assert csv_text[0].startswith("patient_id,")
        assert any("Patient-01" in line and "PR" in line for line in csv_text[1:])
        assert any("Patient-02" in line and "SD" in line for line in csv_text[1:])
