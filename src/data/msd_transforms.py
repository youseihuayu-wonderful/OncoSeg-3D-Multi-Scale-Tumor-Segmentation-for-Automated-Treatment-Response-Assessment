"""MONAI transforms for MSD Task01_BrainTumour (4D NIfTI format).

The MSD format stores all 4 modalities in a single file [H, W, D, 4],
unlike BraTS 2023 which uses separate files per modality.
"""

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


def get_msd_train_transforms(
    roi_size: tuple[int, int, int] = (128, 128, 128),
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Training transforms for MSD 4D NIfTI brain tumor data.

    Pipeline:
        1. Load 4D image [H,W,D,4] and 3D label [H,W,D]
        2. Ensure channel-first: image → [4,H,W,D], label → [1,H,W,D]
        3. Convert integer labels to multi-channel BraTS regions (ET, TC, WT)
        4. Reorient to RAS standard
        5. Resample to isotropic voxel spacing
        6. Z-score normalize each modality channel independently
        7. Crop foreground (remove background air)
        8. Random crop to roi_size
        9. Data augmentation (flip, rotate, intensity)
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_msd_val_transforms(
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Validation/test transforms for MSD format (no augmentation)."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
