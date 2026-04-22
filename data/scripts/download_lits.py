"""Download LiTS (Liver Tumor Segmentation) dataset.

Source: https://competitions.codalab.org/competitions/17094

The LiTS dataset requires registration on CodaLab.

Structure after download:
    lits/
        Training_Batch1/
            volume-0.nii
            segmentation-0.nii
        Training_Batch2/
            volume-27.nii
            segmentation-27.nii
        ...

Labels: 0=Background, 1=Liver, 2=Tumor
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download/verify LiTS dataset")
    parser.add_argument("--output", type=str, default="data/raw/lits")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.verify_only:
        verify_dataset(output_dir)
        return

    print("=" * 60)
    print("LiTS Dataset Download Instructions")
    print("=" * 60)
    print()
    print("The LiTS dataset requires CodaLab registration:")
    print()
    print("1. Go to https://competitions.codalab.org/competitions/17094")
    print("2. Create a CodaLab account and join the competition")
    print("3. Download Training Batch 1 and Training Batch 2")
    print(f"4. Extract to: {output_dir.resolve()}")
    print()
    print("Expected structure:")
    print(f"  {output_dir}/")
    print("  ├── Training_Batch1/")
    print("  │   ├── volume-0.nii")
    print("  │   ├── segmentation-0.nii")
    print("  │   └── ...")
    print("  └── Training_Batch2/")
    print("      ├── volume-27.nii")
    print("      ├── segmentation-27.nii")
    print("      └── ...")
    print()
    print("After downloading, verify with:")
    print(f"  python {__file__} --output {output_dir} --verify-only")
    print()
    print("Note: Total dataset is ~17 GB (131 training subjects).")


def verify_dataset(data_dir: Path):
    """Verify downloaded LiTS dataset."""
    vol_files = sorted(
        list(data_dir.rglob("volume-*.nii.gz"))
        + list(data_dir.rglob("volume-*.nii"))
    )
    seg_files = sorted(
        list(data_dir.rglob("segmentation-*.nii.gz"))
        + list(data_dir.rglob("segmentation-*.nii"))
    )

    # Match pairs
    vol_ids = set()
    for f in vol_files:
        case_id = f.stem.replace("volume-", "").replace(".nii", "")
        vol_ids.add(case_id)

    seg_ids = set()
    for f in seg_files:
        case_id = f.stem.replace("segmentation-", "").replace(".nii", "")
        seg_ids.add(case_id)

    paired = vol_ids & seg_ids

    print("\nLiTS Verification")
    print(f"  Directory: {data_dir}")
    print(f"  Volume files: {len(vol_files)}")
    print(f"  Segmentation files: {len(seg_files)}")
    print(f"  Matched pairs: {len(paired)}")

    if len(paired) >= 130:
        print("  Status: PASSED")
    elif len(paired) > 0:
        print(f"  Status: PARTIAL ({len(paired)} pairs)")
    else:
        print("  Status: FAILED — no valid pairs found")


if __name__ == "__main__":
    main()
