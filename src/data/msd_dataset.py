"""Medical Segmentation Decathlon (MSD) Task01_BrainTumour dataset loader.

Data source: https://medicaldecathlon.com/
Downloaded from: https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar

This dataset contains real clinical brain tumor MRI scans with 4 modalities
(FLAIR, T1w, T1gd, T2w) stored as 4D NIfTI files, and integer label maps.

Labels:
    0 = Background
    1 = Edema
    2 = Non-enhancing tumor
    3 = Enhancing tumor

Evaluation regions (BraTS convention):
    Whole Tumor (WT) = labels 1 + 2 + 3
    Tumor Core (TC)  = labels 2 + 3
    Enhancing Tumor (ET) = label 3
"""

import json
from pathlib import Path

from monai.data import CacheDataset, Dataset
from monai.transforms import Compose


class MSDBrainTumorDataset:
    """MSD Task01_BrainTumour loader for 4D NIfTI volumes.

    The MSD format stores all 4 MRI modalities in a single 4D file
    [H, W, D, 4] rather than separate files per modality.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transform: Compose | None = None,
        cache_rate: float = 0.1,
        val_split: float = 0.2,
        seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.cache_rate = cache_rate
        self.val_split = val_split
        self.seed = seed

        self.metadata = self._load_metadata()
        self.data_dicts = self._build_data_list()

    def _load_metadata(self) -> dict:
        """Load dataset.json for dataset metadata."""
        meta_path = self.root_dir / "dataset.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"dataset.json not found at {meta_path}. "
                f"Make sure the dataset is extracted correctly."
            )
        with open(meta_path) as f:
            return json.load(f)

    def _build_data_list(self) -> list[dict]:
        """Build list of {image, label} dicts from dataset.json."""
        if self.split == "test":
            # Test set has no labels
            entries = self.metadata.get("test", [])
            data_dicts = []
            for entry in entries:
                img_path = self.root_dir / entry
                if img_path.exists():
                    data_dicts.append({"image": str(img_path)})
            return data_dicts

        # Training data — split into train/val
        training_entries = self.metadata.get("training", [])
        data_dicts = []

        for entry in training_entries:
            img_path = self.root_dir / entry["image"]
            lbl_path = self.root_dir / entry["label"]

            if img_path.exists() and lbl_path.exists():
                data_dicts.append(
                    {
                        "image": str(img_path),
                        "label": str(lbl_path),
                    }
                )

        # Deterministic train/val split
        import random

        rng = random.Random(self.seed)
        indices = list(range(len(data_dicts)))
        rng.shuffle(indices)

        n_val = int(len(data_dicts) * self.val_split)

        if self.split == "train":
            selected = [data_dicts[i] for i in indices[n_val:]]
        elif self.split == "val":
            selected = [data_dicts[i] for i in indices[:n_val]]
        else:
            raise ValueError(f"Unknown split: {self.split}. Use 'train', 'val', or 'test'.")

        return selected

    def get_dataset(self) -> Dataset:
        """Return a MONAI Dataset (or CacheDataset) ready for DataLoader."""
        if self.cache_rate > 0 and len(self.data_dicts) > 0:
            return CacheDataset(
                data=self.data_dicts,
                transform=self.transform,
                cache_rate=self.cache_rate,
                num_workers=4,
            )
        return Dataset(data=self.data_dicts, transform=self.transform)

    def __len__(self) -> int:
        return len(self.data_dicts)

    def summary(self) -> str:
        """Print dataset summary."""
        lines = [
            "MSD Brain Tumor Dataset",
            f"  Root: {self.root_dir}",
            f"  Split: {self.split}",
            f"  Subjects: {len(self.data_dicts)}",
            f"  Modalities: {self.metadata.get('modality', {})}",
            f"  Labels: {self.metadata.get('labels', {})}",
        ]
        return "\n".join(lines)
