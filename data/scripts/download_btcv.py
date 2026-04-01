"""Download BTCV (Beyond the Cranial Vault) multi-organ segmentation dataset.

Source: https://www.synapse.org/#!Synapse:syn3193805

The BTCV dataset requires Synapse registration.

Structure after download:
    btcv/
        Training/
            img/
                img0001.nii.gz
                img0002.nii.gz
            label/
                label0001.nii.gz
                label0002.nii.gz

Labels (13 organs):
    0=Background, 1=Spleen, 2=Right Kidney, 3=Left Kidney,
    4=Gallbladder, 5=Esophagus, 6=Liver, 7=Stomach, 8=Aorta,
    9=IVC, 10=Portal/Splenic Vein, 11=Pancreas,
    12=Right Adrenal, 13=Left Adrenal
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download/verify BTCV dataset")
    parser.add_argument("--output", type=str, default="data/raw/btcv")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.verify_only:
        verify_dataset(output_dir)
        return

    print("=" * 60)
    print("BTCV Dataset Download Instructions")
    print("=" * 60)
    print()
    print("The BTCV dataset requires Synapse registration:")
    print()
    print("1. Go to https://www.synapse.org/#!Synapse:syn3193805")
    print("2. Create a Synapse account (free)")
    print("3. Download 'Abdomen/RawData.zip'")
    print(f"4. Extract to: {output_dir.resolve()}")
    print()
    print("Expected structure:")
    print(f"  {output_dir}/")
    print(f"  ├── Training/")
    print(f"  │   ├── img/")
    print(f"  │   │   ├── img0001.nii.gz")
    print(f"  │   │   └── ...")
    print(f"  │   └── label/")
    print(f"  │       ├── label0001.nii.gz")
    print(f"  │       └── ...")
    print()
    print("After downloading, verify with:")
    print(f"  python {__file__} --output {output_dir} --verify-only")
    print()
    print("Note: 30 training subjects, ~1.5 GB total. 13 organ labels.")


def verify_dataset(data_dir: Path):
    """Verify downloaded BTCV dataset."""
    # Try common structures
    img_dir = None
    for candidate in [
        data_dir / "Training" / "img",
        data_dir / "img",
        data_dir / "imagesTr",
    ]:
        if candidate.is_dir():
            img_dir = candidate
            break

    lbl_dir = None
    for candidate in [
        data_dir / "Training" / "label",
        data_dir / "label",
        data_dir / "labelsTr",
    ]:
        if candidate.is_dir():
            lbl_dir = candidate
            break

    if img_dir is None:
        print(f"ERROR: No image directory found in {data_dir}")
        return

    img_files = sorted(img_dir.glob("img*.nii.gz"))
    lbl_files = sorted(lbl_dir.glob("label*.nii.gz")) if lbl_dir else []

    # Match pairs
    img_ids = {f.stem.replace("img", "") for f in img_files}
    lbl_ids = {f.stem.replace(".nii", "").replace("label", "") for f in lbl_files}
    paired = img_ids & lbl_ids

    print(f"\nBTCV Verification")
    print(f"  Image dir: {img_dir}")
    print(f"  Label dir: {lbl_dir}")
    print(f"  Image files: {len(img_files)}")
    print(f"  Label files: {len(lbl_files)}")
    print(f"  Matched pairs: {len(paired)}")

    if len(paired) >= 24:
        print("  Status: PASSED")
    elif len(paired) > 0:
        print(f"  Status: PARTIAL ({len(paired)} pairs)")
    else:
        print("  Status: FAILED — no valid pairs found")


if __name__ == "__main__":
    main()
