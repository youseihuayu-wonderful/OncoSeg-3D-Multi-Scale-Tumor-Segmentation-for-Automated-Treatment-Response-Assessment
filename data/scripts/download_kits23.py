"""Download KiTS23 (Kidney Tumor Segmentation) dataset.

Source: https://kits-challenge.org/kits23/
GitHub: https://github.com/neheller/kits23

Requires: pip install git+https://github.com/neheller/kits23

Structure after download:
    kits23/dataset/
        case_00000/
            imaging.nii.gz
            segmentation.nii.gz
        case_00001/
        ...

Labels: 0=Background, 1=Kidney, 2=Tumor, 3=Cyst
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download KiTS23 dataset")
    parser.add_argument("--output", type=str, default="data/raw/kits23")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.verify_only:
        verify_dataset(output_dir)
        return

    print("=" * 60)
    print("KiTS23 Dataset Download")
    print("=" * 60)
    print()
    print("Option 1: Automated download via kits23 Python package")
    print()
    print("  pip install git+https://github.com/neheller/kits23")
    print(f"  python -m kits23.entry --output {output_dir}")
    print()
    print("Option 2: Clone and run manually")
    print()
    print("  git clone https://github.com/neheller/kits23.git")
    print("  cd kits23")
    print("  pip install -e .")
    print("  python -m kits23.entry")
    print()
    print("After downloading, verify with:")
    print(f"  python {__file__} --output {output_dir} --verify-only")
    print()
    print("Note: Full dataset is ~30 GB. Download takes 1-2 hours.")


def verify_dataset(data_dir: Path):
    """Verify downloaded KiTS23 dataset."""
    dataset_dir = data_dir
    if (data_dir / "dataset").is_dir():
        dataset_dir = data_dir / "dataset"

    cases = sorted([d for d in dataset_dir.iterdir()
                    if d.is_dir() and d.name.startswith("case_")])

    complete = 0
    for case in cases:
        img = case / "imaging.nii.gz"
        seg = case / "segmentation.nii.gz"
        if img.exists() and seg.exists():
            complete += 1

    print(f"\nKiTS23 Verification")
    print(f"  Directory: {dataset_dir}")
    print(f"  Cases found: {len(cases)}")
    print(f"  Complete (img+seg): {complete}")
    print(f"  Missing segmentation: {len(cases) - complete}")

    if complete > 400:
        print("  Status: PASSED")
    elif complete > 0:
        print("  Status: PARTIAL (some cases missing segmentation — this is expected for test cases)")
    else:
        print("  Status: FAILED — no valid cases found")


if __name__ == "__main__":
    main()
