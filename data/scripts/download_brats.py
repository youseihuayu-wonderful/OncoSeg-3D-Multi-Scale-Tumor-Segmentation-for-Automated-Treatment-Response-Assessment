"""Download BraTS 2023 dataset.

The BraTS dataset requires registration at https://www.synapse.org/brats
This script provides instructions and verifies the downloaded data.
"""

import argparse
from pathlib import Path


def verify_dataset(data_dir: Path) -> dict:
    """Verify downloaded BraTS dataset structure."""
    stats = {"subjects": 0, "modalities": set(), "complete": 0, "incomplete": 0}

    expected_suffixes = ["t1n", "t1c", "t2w", "t2f", "seg"]

    for subject_dir in sorted(data_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        stats["subjects"] += 1
        found_suffixes = []

        for f in subject_dir.glob("*.nii.gz"):
            for suffix in expected_suffixes:
                if suffix in f.name:
                    found_suffixes.append(suffix)
                    stats["modalities"].add(suffix)

        if set(found_suffixes) >= set(expected_suffixes):
            stats["complete"] += 1
        else:
            stats["incomplete"] += 1
            missing = set(expected_suffixes) - set(found_suffixes)
            print(f"  WARNING: {subject_dir.name} missing: {missing}")

    stats["modalities"] = sorted(stats["modalities"])
    return stats


def main():
    parser = argparse.ArgumentParser(description="Download/verify BraTS 2023 dataset")
    parser.add_argument("--output", type=str, default="data/raw/brats2023")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.verify_only:
        if not output_dir.exists():
            print(f"ERROR: Directory not found: {output_dir}")
            return

        for split in ["train", "val", "test"]:
            split_dir = output_dir / split
            if split_dir.exists():
                stats = verify_dataset(split_dir)
                print(f"\n{split.upper()} split:")
                print(f"  Subjects: {stats['subjects']}")
                print(f"  Complete: {stats['complete']}")
                print(f"  Incomplete: {stats['incomplete']}")
                print(f"  Modalities found: {stats['modalities']}")
        return

    print("=" * 60)
    print("BraTS 2023 Dataset Download Instructions")
    print("=" * 60)
    print()
    print("The BraTS dataset requires manual registration:")
    print()
    print("1. Go to https://www.synapse.org/brats")
    print("2. Create a Synapse account (free)")
    print("3. Join the BraTS 2023 challenge")
    print("4. Download the training data")
    print(f"5. Extract to: {output_dir.resolve()}")
    print()
    print("Expected directory structure:")
    print(f"  {output_dir}/")
    print("  ├── train/")
    print("  │   ├── BraTS-GLI-00000-000/")
    print("  │   │   ├── *-t1n.nii.gz")
    print("  │   │   ├── *-t1c.nii.gz")
    print("  │   │   ├── *-t2w.nii.gz")
    print("  │   │   ├── *-t2f.nii.gz")
    print("  │   │   └── *-seg.nii.gz")
    print("  │   └── ...")
    print("  └── val/")
    print()
    print("After downloading, verify with:")
    print(f"  python {__file__} --output {output_dir} --verify-only")


if __name__ == "__main__":
    main()
