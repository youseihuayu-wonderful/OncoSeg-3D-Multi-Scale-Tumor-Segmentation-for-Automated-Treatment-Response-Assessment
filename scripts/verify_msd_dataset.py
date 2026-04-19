"""Pre-flight integrity check for the MSD Task01_BrainTumour dataset.

Run this BEFORE starting a multi-hour training run on Kaggle/Colab to
catch bad downloads, truncated NIfTI files, image/label shape
mismatches, or invalid label values — any of which would crash
training minutes into the run and waste the GPU budget.

Usage:
    python scripts/verify_msd_dataset.py /path/to/Task01_BrainTumour
    python scripts/verify_msd_dataset.py --dataset-root /kaggle/working/data/Task01_BrainTumour

Exit codes:
    0 — all checks pass
    1 — one or more checks failed (report printed to stderr)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

# MSD Task01 spec: 4 modalities (T1, T1c, T2, FLAIR), 4 label classes (0-3).
EXPECTED_MODALITIES = 4
EXPECTED_LABEL_VALUES = {0, 1, 2, 3}
MIN_TRAINING_IMAGES = 380  # MSD publishes 484 total; filterable count ~388.
MIN_SPATIAL_SIZE = 32  # anything smaller is a corruption


@dataclass
class Report:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def ok(self, msg: str) -> None:
        self.passed.append(msg)

    def fail(self, msg: str) -> None:
        self.failed.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        lines = [
            f"Passed:   {len(self.passed)}",
            f"Warnings: {len(self.warnings)}",
            f"Failed:   {len(self.failed)}",
        ]
        if self.warnings:
            lines.append("\n-- Warnings --")
            lines.extend(f"  ! {w}" for w in self.warnings)
        if self.failed:
            lines.append("\n-- Failures --")
            lines.extend(f"  X {f}" for f in self.failed)
        return "\n".join(lines)


def check_directory_structure(root: Path, report: Report) -> bool:
    if not root.exists():
        report.fail(f"Dataset root does not exist: {root}")
        return False
    required = ["imagesTr", "labelsTr", "dataset.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        report.fail(f"Missing required entries in {root}: {missing}")
        return False
    report.ok(f"Directory structure present at {root}")
    return True


def load_dataset_json(root: Path, report: Report) -> dict | None:
    path = root / "dataset.json"
    try:
        meta = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        report.fail(f"Cannot parse dataset.json: {e}")
        return None

    for key in ("name", "numTraining", "training"):
        if key not in meta:
            report.fail(f"dataset.json missing required key: {key}")
            return None

    if meta["name"] != "BRATS":
        report.warn(
            f"dataset.json 'name' = {meta['name']!r}, expected 'BRATS'"
        )

    n_listed = len(meta["training"])
    n_claimed = meta["numTraining"]
    if n_listed != n_claimed:
        report.fail(
            f"dataset.json inconsistency: numTraining={n_claimed} but "
            f"training list has {n_listed} entries"
        )
        return None
    report.ok(f"dataset.json parsed, {n_listed} training entries listed")
    return meta


def check_file_count(root: Path, meta: dict, report: Report) -> None:
    images = sorted((root / "imagesTr").glob("BRATS_*.nii.gz"))
    labels = sorted((root / "labelsTr").glob("BRATS_*.nii.gz"))
    if len(images) < MIN_TRAINING_IMAGES:
        report.fail(
            f"Only {len(images)} training images found under imagesTr/ "
            f"(expected >= {MIN_TRAINING_IMAGES}). Download may be truncated."
        )
    else:
        report.ok(f"Found {len(images)} training images")
    if len(labels) != len(images):
        report.fail(
            f"Image/label count mismatch: {len(images)} images vs {len(labels)} labels"
        )
    else:
        report.ok(f"Image/label count matches ({len(labels)} each)")


def check_nifti_sample(
    root: Path, meta: dict, report: Report, n_sample: int = 3
) -> None:
    """Open the first N training pairs, check shape + modality count + labels."""
    entries = meta["training"][:n_sample]
    for entry in entries:
        img_rel = entry["image"].lstrip("./")
        lbl_rel = entry["label"].lstrip("./")
        img_path = root / img_rel
        lbl_path = root / lbl_rel

        if not img_path.exists():
            report.fail(f"Missing image file: {img_path}")
            continue
        if not lbl_path.exists():
            report.fail(f"Missing label file: {lbl_path}")
            continue

        try:
            img = nib.load(str(img_path))
            lbl = nib.load(str(lbl_path))
        except Exception as e:
            report.fail(f"Cannot load NIfTI {img_path.name}: {type(e).__name__}: {e}")
            continue

        img_shape = img.shape
        lbl_shape = lbl.shape

        if len(img_shape) != 4 or img_shape[-1] != EXPECTED_MODALITIES:
            report.fail(
                f"{img_path.name}: expected 4D with {EXPECTED_MODALITIES} "
                f"modalities, got shape {img_shape}"
            )
            continue

        spatial = img_shape[:3]
        if any(s < MIN_SPATIAL_SIZE for s in spatial):
            report.fail(
                f"{img_path.name}: spatial size {spatial} below minimum "
                f"{MIN_SPATIAL_SIZE} (corrupt file?)"
            )
            continue

        if spatial != lbl_shape[:3]:
            report.fail(
                f"{img_path.name}: image spatial {spatial} does not match "
                f"label spatial {lbl_shape[:3]}"
            )
            continue

        # Spot-check label values (load full array for small images; fine for 3 samples)
        lbl_data = np.asarray(lbl.dataobj)
        unique = set(np.unique(lbl_data).tolist())
        unknown = unique - EXPECTED_LABEL_VALUES
        if unknown:
            report.fail(
                f"{lbl_path.name}: unexpected label values {unknown} "
                f"(expected subset of {sorted(EXPECTED_LABEL_VALUES)})"
            )
            continue

        report.ok(
            f"{img_path.name}: shape OK ({spatial}, 4 modalities); "
            f"labels OK ({sorted(unique)})"
        )


def run_all_checks(root: Path) -> Report:
    report = Report()
    if not check_directory_structure(root, report):
        return report
    meta = load_dataset_json(root, report)
    if meta is None:
        return report
    check_file_count(root, meta, report)
    check_nifti_sample(root, meta, report, n_sample=3)
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-flight integrity check for MSD Task01_BrainTumour."
    )
    p.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        help="Path to Task01_BrainTumour directory",
    )
    p.add_argument(
        "--dataset-root",
        dest="dataset_root_flag",
        type=Path,
        help="Alternative way to specify dataset root",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.dataset_root or args.dataset_root_flag
    if root is None:
        print(
            "error: dataset_root is required "
            "(positional or via --dataset-root)",
            file=sys.stderr,
        )
        return 2

    report = run_all_checks(root)
    print(report.summary())
    if report.failed:
        print(
            f"\nRESULT: FAIL ({len(report.failed)} issues). "
            "Do not start training — fix the dataset first.",
            file=sys.stderr,
        )
        return 1
    print("\nRESULT: PASS. Dataset looks good for training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
