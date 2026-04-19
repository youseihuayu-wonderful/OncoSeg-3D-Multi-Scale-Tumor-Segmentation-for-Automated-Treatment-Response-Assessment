"""Tests for scripts/verify_msd_dataset.py using tiny synthetic fixtures.

We build a miniature MSD-like directory inside tmp_path so we can exercise
every pass/fail branch without the real 7 GB dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.verify_msd_dataset import (
    EXPECTED_MODALITIES,
    MIN_TRAINING_IMAGES,
    main,
    run_all_checks,
)


def _write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, str(path))


def _build_valid_dataset(root: Path, n: int = MIN_TRAINING_IMAGES) -> None:
    """Create n training pairs with valid shapes + label values."""
    training = []
    for i in range(n):
        subj = f"BRATS_{i:03d}"
        img_rel = f"./imagesTr/{subj}.nii.gz"
        lbl_rel = f"./labelsTr/{subj}.nii.gz"
        training.append({"image": img_rel, "label": lbl_rel})
        # Only materialise NIfTI files for the first few — the script
        # only opens the first N as samples. Full-count scan is file
        # listing only, which just needs the files to exist on disk.
        if i < 5:
            img_data = np.random.RandomState(i).rand(40, 40, 40, 4).astype(np.float32)
            lbl_data = np.random.RandomState(i).randint(0, 4, size=(40, 40, 40)).astype(np.int16)
            _write_nifti(root / img_rel.lstrip("./"), img_data)
            _write_nifti(root / lbl_rel.lstrip("./"), lbl_data)
        else:
            # Create empty stub files so the count-check passes without
            # needing hundreds of real NIfTI writes.
            (root / img_rel.lstrip("./")).parent.mkdir(parents=True, exist_ok=True)
            (root / lbl_rel.lstrip("./")).parent.mkdir(parents=True, exist_ok=True)
            (root / img_rel.lstrip("./")).touch()
            (root / lbl_rel.lstrip("./")).touch()
    (root / "dataset.json").write_text(
        json.dumps({"name": "BRATS", "numTraining": n, "training": training})
    )


class TestValidDataset:
    @pytest.fixture
    def dataset_root(self, tmp_path: Path) -> Path:
        root = tmp_path / "Task01_BrainTumour"
        _build_valid_dataset(root)
        return root

    def test_run_all_checks_passes(self, dataset_root: Path):
        report = run_all_checks(dataset_root)
        assert report.failed == [], f"Unexpected failures: {report.failed}"
        assert len(report.passed) >= 4, "Expected at least structure/json/count/sample checks"

    def test_main_returns_zero(self, dataset_root: Path, capsys):
        assert main([str(dataset_root)]) == 0
        captured = capsys.readouterr()
        assert "PASS" in captured.out


class TestFailurePaths:
    def test_missing_root(self, tmp_path: Path, capsys):
        missing = tmp_path / "does_not_exist"
        report = run_all_checks(missing)
        assert any("does not exist" in f for f in report.failed)

    def test_missing_dataset_json(self, tmp_path: Path):
        root = tmp_path / "ds"
        (root / "imagesTr").mkdir(parents=True)
        (root / "labelsTr").mkdir(parents=True)
        report = run_all_checks(root)
        assert any("dataset.json" in f for f in report.failed)

    def test_corrupt_dataset_json(self, tmp_path: Path):
        root = tmp_path / "ds"
        _build_valid_dataset(root)
        (root / "dataset.json").write_text("not json at all {{{")
        report = run_all_checks(root)
        assert any("Cannot parse dataset.json" in f for f in report.failed)

    def test_num_training_mismatch(self, tmp_path: Path):
        root = tmp_path / "ds"
        _build_valid_dataset(root)
        meta = json.loads((root / "dataset.json").read_text())
        meta["numTraining"] = 999  # lie about the count
        (root / "dataset.json").write_text(json.dumps(meta))
        report = run_all_checks(root)
        assert any("numTraining" in f for f in report.failed)

    def test_too_few_training_images(self, tmp_path: Path):
        root = tmp_path / "ds"
        _build_valid_dataset(root, n=10)  # well below MIN_TRAINING_IMAGES
        report = run_all_checks(root)
        assert any(str(MIN_TRAINING_IMAGES) in f or "truncated" in f.lower()
                   for f in report.failed)

    def test_bad_label_values(self, tmp_path: Path):
        root = tmp_path / "ds"
        _build_valid_dataset(root)
        # Replace first label file with out-of-range values
        bad = np.full((40, 40, 40), 99, dtype=np.int16)
        _write_nifti(root / "labelsTr/BRATS_000.nii.gz", bad)
        report = run_all_checks(root)
        assert any("unexpected label values" in f for f in report.failed)

    def test_shape_mismatch(self, tmp_path: Path):
        root = tmp_path / "ds"
        _build_valid_dataset(root)
        # Replace first image with wrong modality count
        bad = np.random.rand(40, 40, 40, 2).astype(np.float32)  # only 2 modalities
        _write_nifti(root / "imagesTr/BRATS_000.nii.gz", bad)
        report = run_all_checks(root)
        assert any(f"{EXPECTED_MODALITIES} modalities" in f for f in report.failed)

    def test_image_label_spatial_mismatch(self, tmp_path: Path):
        root = tmp_path / "ds"
        _build_valid_dataset(root)
        # Replace image with smaller spatial size
        bad = np.random.rand(32, 32, 32, 4).astype(np.float32)
        _write_nifti(root / "imagesTr/BRATS_000.nii.gz", bad)
        report = run_all_checks(root)
        assert any("does not match" in f and "label spatial" in f for f in report.failed)


class TestCLI:
    def test_no_args_errors(self, capsys):
        assert main([]) == 2

    def test_flag_form(self, tmp_path: Path):
        root = tmp_path / "ds"
        _build_valid_dataset(root)
        assert main(["--dataset-root", str(root)]) == 0

    def test_fail_exit_code(self, tmp_path: Path):
        assert main([str(tmp_path / "missing")]) == 1
