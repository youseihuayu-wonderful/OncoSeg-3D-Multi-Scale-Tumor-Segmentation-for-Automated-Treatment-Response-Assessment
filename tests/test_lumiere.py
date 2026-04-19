"""Unit tests for the LUMIERE dataset loader."""

import nibabel as nib
import numpy as np
import pytest

from src.data.lumiere import (
    ONCOSEG_MODALITIES,
    discover_lumiere,
    flatten_timepoints,
    patients_with_followup,
)

MODALITY_FILENAMES = {
    "t1n": "t1_skull_strip.nii.gz",
    "t1c": "ct1_skull_strip.nii.gz",
    "t2w": "t2_skull_strip.nii.gz",
    "t2f": "flair_skull_strip.nii.gz",
}


def _write_nii(path, shape=(4, 4, 4)):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), str(path))


def _make_week(patient_dir, week_idx, modalities=ONCOSEG_MODALITIES, with_seg=True):
    week_dir = patient_dir / f"week-{week_idx:03d}"
    strip_dir = week_dir / "DeepBraTumIA-segmentation/atlas/skull_strip"
    for mod in modalities:
        _write_nii(strip_dir / MODALITY_FILENAMES[mod])
    if with_seg:
        _write_nii(week_dir / "DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii.gz")
        _write_nii(week_dir / "HD-GLIO-AUTO-segmentation/segmentation.nii.gz")
    return week_dir


def _write_clinical_csv(root, rows):
    path = root / "LumiereClinicalData.csv"
    header = "patient_id,week,rano,rationale\n"
    body = "\n".join(f"{p},{w},{r},{rat}" for p, w, r, rat in rows)
    path.write_text(header + body + "\n")
    return path


class TestDiscoverLumiere:
    """Scan a synthetic LUMIERE layout and validate loader output."""

    def test_discovers_patients_and_timepoints(self, tmp_path):
        p1 = tmp_path / "Patient-01"
        _make_week(p1, 0)
        _make_week(p1, 13)
        _make_week(p1, 24)

        p2 = tmp_path / "Patient-02"
        _make_week(p2, 0)
        _make_week(p2, 8)

        patients = discover_lumiere(tmp_path)

        assert [p.patient_id for p in patients] == ["Patient-01", "Patient-02"]
        assert [len(p.timepoints) for p in patients] == [3, 2]

    def test_timepoints_are_sorted_by_week(self, tmp_path):
        p = tmp_path / "Patient-01"
        _make_week(p, 24)
        _make_week(p, 0)
        _make_week(p, 13)

        patients = discover_lumiere(tmp_path)
        weeks = [tp.week for tp in patients[0].timepoints]
        assert weeks == [0, 13, 24]

    def test_skips_timepoints_missing_modalities(self, tmp_path):
        p = tmp_path / "Patient-01"
        _make_week(p, 0)
        # Week 13 is missing FLAIR — must be skipped
        _make_week(p, 13, modalities=("t1n", "t1c", "t2w"))

        patients = discover_lumiere(tmp_path)

        assert len(patients) == 1
        assert [tp.week for tp in patients[0].timepoints] == [0]

    def test_skips_non_patient_directories(self, tmp_path):
        (tmp_path / "scripts").mkdir()
        (tmp_path / "README").mkdir()
        _make_week(tmp_path / "Patient-01", 0)

        patients = discover_lumiere(tmp_path)
        assert len(patients) == 1

    def test_data_dict_contains_all_modalities(self, tmp_path):
        _make_week(tmp_path / "Patient-01", 0)

        patients = discover_lumiere(tmp_path)
        entry = patients[0].timepoints[0].as_data_dict()

        for mod in ONCOSEG_MODALITIES:
            assert mod in entry
            assert entry[mod].endswith(".nii.gz")
        assert entry["subject_id"] == "Patient-01__week-000"

    def test_detects_deepbratumia_and_hdglio_segs(self, tmp_path):
        _make_week(tmp_path / "Patient-01", 0, with_seg=True)

        patients = discover_lumiere(tmp_path)
        tp = patients[0].timepoints[0]

        assert tp.deepbratumia_seg is not None
        assert tp.hdglio_seg is not None

    def test_missing_segs_are_none(self, tmp_path):
        _make_week(tmp_path / "Patient-01", 0, with_seg=False)

        patients = discover_lumiere(tmp_path)
        tp = patients[0].timepoints[0]

        assert tp.deepbratumia_seg is None
        assert tp.hdglio_seg is None

    def test_loads_rano_from_csv(self, tmp_path):
        _make_week(tmp_path / "Patient-01", 0)
        _make_week(tmp_path / "Patient-01", 13)
        _write_clinical_csv(
            tmp_path,
            [
                ("Patient-01", 0, "baseline", "initial scan"),
                ("Patient-01", 13, "PR", "30% decrease in enhancement"),
            ],
        )

        patients = discover_lumiere(tmp_path)
        tps = patients[0].timepoints

        assert tps[0].rano == "baseline"
        assert tps[1].rano == "PR"
        assert "decrease" in tps[1].rano_rationale

    def test_missing_csv_does_not_fail(self, tmp_path):
        _make_week(tmp_path / "Patient-01", 0)

        patients = discover_lumiere(tmp_path)
        assert patients[0].timepoints[0].rano is None

    def test_raises_when_root_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            discover_lumiere(tmp_path / "does-not-exist")


class TestFilters:
    def test_patients_with_followup_filters_singletons(self, tmp_path):
        _make_week(tmp_path / "Patient-01", 0)  # only baseline
        _make_week(tmp_path / "Patient-02", 0)
        _make_week(tmp_path / "Patient-02", 13)

        patients = discover_lumiere(tmp_path)
        with_fu = patients_with_followup(patients)

        assert [p.patient_id for p in with_fu] == ["Patient-02"]

    def test_flatten_timepoints_preserves_order(self, tmp_path):
        _make_week(tmp_path / "Patient-01", 0)
        _make_week(tmp_path / "Patient-01", 13)
        _make_week(tmp_path / "Patient-02", 0)

        patients = discover_lumiere(tmp_path)
        flat = flatten_timepoints(patients)

        assert [(tp.patient_id, tp.week) for tp in flat] == [
            ("Patient-01", 0),
            ("Patient-01", 13),
            ("Patient-02", 0),
        ]


class TestPatientHelpers:
    def test_baseline_and_followups(self, tmp_path):
        _make_week(tmp_path / "Patient-01", 0)
        _make_week(tmp_path / "Patient-01", 13)
        _make_week(tmp_path / "Patient-01", 24)

        patients = discover_lumiere(tmp_path)
        p = patients[0]

        assert p.baseline().week == 0
        assert [fu.week for fu in p.followups()] == [13, 24]

    def test_baseline_empty_patient(self):
        from src.data.lumiere import LumierePatient

        p = LumierePatient(patient_id="Patient-99")
        assert p.baseline() is None
        assert p.followups() == []
