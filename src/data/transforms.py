"""MONAI transforms for 3D medical image preprocessing and augmentation."""

from monai.transforms import (
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
)


def get_train_transforms(
    roi_size: tuple[int, int, int] = (128, 128, 128),
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Training transforms with data augmentation.

    Pipeline:
        1. Load NIfTI files
        2. Reorient to RAS
        3. Resample to isotropic spacing
        4. Normalize intensity (per-channel z-score)
        5. Crop foreground (remove background air)
        6. Random spatial crop to roi_size
        7. Random augmentations (flip, rotate, intensity)
    """
    keys = ["t1n", "t1c", "t2w", "t2f", "label"]
    image_keys = ["t1n", "t1c", "t2w", "t2f"]

    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear",) * 4 + ("nearest",)),
            NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
            CropForegroundd(keys=keys, source_key="t1c"),
            RandSpatialCropd(keys=keys, roi_size=roi_size, random_size=False),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=keys, prob=0.5, max_k=3),
            RandScaleIntensityd(keys=image_keys, factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=image_keys, offsets=0.1, prob=0.5),
            EnsureTyped(keys=keys),
        ]
    )


def get_val_transforms(
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Validation/test transforms (no augmentation)."""
    keys = ["t1n", "t1c", "t2w", "t2f", "label"]
    image_keys = ["t1n", "t1c", "t2w", "t2f"]

    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear",) * 4 + ("nearest",)),
            NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
            CropForegroundd(keys=keys, source_key="t1c"),
            EnsureTyped(keys=keys),
        ]
    )
