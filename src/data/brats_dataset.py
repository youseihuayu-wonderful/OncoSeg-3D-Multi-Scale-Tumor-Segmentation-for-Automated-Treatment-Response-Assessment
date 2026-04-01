"""BraTS 2023 dataset loader for brain tumor segmentation."""

from pathlib import Path

from monai.data import CacheDataset, Dataset
from monai.transforms import Compose


class BraTSDataset:
    """BraTS 2023 dataset with 4 MRI modalities (T1, T1ce, T2, FLAIR).

    Labels:
        0 = Background
        1 = Necrotic / Non-Enhancing Tumor Core (NCR/NET)
        2 = Peritumoral Edema (ED)
        3 = GD-Enhancing Tumor (ET)

    Evaluation regions:
        Whole Tumor (WT) = labels 1 + 2 + 3
        Tumor Core (TC)  = labels 1 + 3
        Enhancing Tumor (ET) = label 3
    """

    # BraTS modality file suffixes
    MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
    LABEL_SUFFIX = "seg"

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transform: Compose | None = None,
        cache_rate: float = 0.1,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.cache_rate = cache_rate

        self.data_dicts = self._build_data_list()

    def _build_data_list(self) -> list[dict]:
        """Scan directory for BraTS subject folders and build data dicts."""
        data_dicts = []
        split_dir = self.root_dir / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for subject_dir in sorted(split_dir.iterdir()):
            if not subject_dir.is_dir():
                continue

            entry = {}
            # Find modality files
            for mod in self.MODALITIES:
                matches = list(subject_dir.glob(f"*{mod}.nii.gz"))
                if matches:
                    entry[mod] = str(matches[0])

            # Find label file
            label_matches = list(subject_dir.glob(f"*{self.LABEL_SUFFIX}.nii.gz"))
            if label_matches:
                entry["label"] = str(label_matches[0])

            if len(entry) == len(self.MODALITIES) + 1:
                data_dicts.append(entry)

        return data_dicts

    def get_dataset(self) -> Dataset:
        """Return a MONAI Dataset (or CacheDataset) ready for DataLoader."""
        if self.cache_rate > 0:
            return CacheDataset(
                data=self.data_dicts,
                transform=self.transform,
                cache_rate=self.cache_rate,
                num_workers=4,
            )
        return Dataset(data=self.data_dicts, transform=self.transform)

    def __len__(self) -> int:
        return len(self.data_dicts)
