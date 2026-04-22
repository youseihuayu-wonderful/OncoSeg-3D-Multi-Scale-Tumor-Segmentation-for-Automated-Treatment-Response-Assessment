"""Service layer: wires preprocessing, Predictor, and RECIST measurement together."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import nibabel as nib
import numpy as np
import torch

from src.inference import Predictor, build_inference_transforms
from src.response.classifier import ResponseClassifier
from src.response.recist import RECISTMeasurer

MODALITIES = ("t1n", "t1c", "t2w", "t2f")
CHANNEL_NAMES = ("TC", "WT", "ET")


class PredictorLike(Protocol):
    """Minimal surface the service needs from a predictor (allows test stubs)."""

    def predict_volume(self, image: torch.Tensor) -> dict[str, np.ndarray]: ...


@dataclass
class ServiceMeta:
    model_name: str
    model_source: str
    checkpoint: str | None
    device: str
    roi_size: tuple[int, int, int]
    num_classes: int
    mc_samples: int
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ready: bool = True


@dataclass
class OncoSegService:
    """Stateless request-scoped wrapper around a Predictor.

    Safe to share across workers only if the underlying predictor is
    thread-safe (PyTorch eval-mode forward passes are, provided no autograd).
    """

    predictor: PredictorLike
    meta: ServiceMeta
    measurer: RECISTMeasurer = field(default_factory=RECISTMeasurer)
    classifier: ResponseClassifier = field(default_factory=ResponseClassifier)

    def segment(
        self,
        modality_bytes: dict[str, bytes],
        subject_id: str = "subject",
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
        """Run inference on 4 uploaded NIfTI byte blobs.

        Returns the segmentation [3, H, W, D], the reference affine [4, 4],
        and voxel spacing (pixdim) extracted from the t1c reference.
        """
        self._require_all_modalities(modality_bytes)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dict: dict[str, str] = {"subject_id": subject_id}
            for mod, blob in modality_bytes.items():
                dest = tmp_path / f"{mod}.nii.gz"
                dest.write_bytes(blob)
                data_dict[mod] = str(dest)

            transforms = build_inference_transforms(pixdim=self.meta.pixdim)
            processed = transforms(data_dict)
            image = torch.cat(
                [processed[k] for k in MODALITIES], dim=0
            ).unsqueeze(0)

            result = self.predictor.predict_volume(image)

            ref = nib.load(data_dict["t1c"])
            affine = np.asarray(ref.affine)
            pixdim = tuple(float(x) for x in ref.header.get_zooms()[:3])

        segmentation = np.asarray(result["segmentation"], dtype=np.uint8)
        if segmentation.ndim != 4 or segmentation.shape[0] != self.meta.num_classes:
            raise ValueError(
                f"Predictor returned unexpected segmentation shape "
                f"{segmentation.shape}; expected [{self.meta.num_classes}, H, W, D]"
            )
        return segmentation, affine, pixdim

    def measure(
        self,
        modality_bytes: dict[str, bytes],
        subject_id: str = "subject",
    ) -> dict:
        segmentation, _, pixdim = self.segment(modality_bytes, subject_id=subject_id)

        channel_stats = []
        for idx, name in enumerate(CHANNEL_NAMES):
            mask = segmentation[idx]
            voxel_count = int(mask.sum())
            volume = float(voxel_count * pixdim[0] * pixdim[1] * pixdim[2])
            channel_stats.append(
                {"name": name, "positive_voxels": voxel_count, "volume_mm3": volume}
            )

        et_mask = segmentation[2].astype(np.uint8)
        lesions = self.measurer.measure_lesions(et_mask, pixdim)
        recist = {
            "num_lesions": len(lesions),
            "sum_longest_diameter_mm": float(
                sum(les["longest_diameter_mm"] for les in lesions)
            ),
            "total_volume_mm3": float(sum(les["volume_mm3"] for les in lesions)),
            "lesions": lesions,
        }

        spatial_shape = tuple(int(s) for s in segmentation.shape[1:])
        return {
            "subject_id": subject_id,
            "shape": spatial_shape,
            "pixdim": pixdim,
            "channel_stats": channel_stats,
            "recist": recist,
        }

    def response(
        self,
        baseline_bytes: dict[str, bytes],
        followup_bytes: dict[str, bytes],
        subject_id: str = "subject",
    ) -> dict:
        baseline_seg, _, baseline_px = self.segment(
            baseline_bytes, subject_id=f"{subject_id}_baseline"
        )
        followup_seg, _, _ = self.segment(
            followup_bytes, subject_id=f"{subject_id}_followup"
        )
        baseline_et = baseline_seg[2].astype(np.uint8)
        followup_et = followup_seg[2].astype(np.uint8)

        result = self.classifier.classify(baseline_et, followup_et, pixdim=baseline_px)
        return {
            "category": result.category.name,
            "category_full": result.category.value,
            "percent_change": float(result.percent_change),
            "baseline_sum_ld_mm": float(result.baseline_sum_ld),
            "followup_sum_ld_mm": float(result.followup_sum_ld),
            "baseline_volume_mm3": float(result.baseline_volume),
            "followup_volume_mm3": float(result.followup_volume),
            "num_baseline_lesions": int(result.num_baseline_lesions),
            "num_followup_lesions": int(result.num_followup_lesions),
        }

    def write_segmentation_nifti(
        self, segmentation: np.ndarray, affine: np.ndarray, dest: Path
    ) -> Path:
        """Persist seg as a 4D uint8 NIfTI (channels-last per NIfTI convention)."""
        channels_last = np.transpose(segmentation, (1, 2, 3, 0)).astype(np.uint8)
        nib.save(nib.Nifti1Image(channels_last, affine), dest)
        return dest

    @staticmethod
    def _require_all_modalities(modality_bytes: dict[str, bytes]) -> None:
        missing = [m for m in MODALITIES if m not in modality_bytes]
        if missing:
            raise ValueError(f"Missing required modalities: {missing}")


def build_predictor_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    roi_size: tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 2,
    mc_samples: int = 0,
    model_source: str = "train_all",
) -> tuple[Predictor, str]:
    """Build a Predictor around a checkpoint produced by train_all.py or src.models.

    `model_source`:
        - "train_all": checkpoint produced by train_all.OncoSeg (inline class).
          Matches the local_results/oncoseg_best.pth layout.
        - "src": checkpoint matching src.models.oncoseg.OncoSeg.
    """
    if model_source == "train_all":
        # Deferred import: train_all pulls heavy deps (matplotlib, MONAI transforms)
        # that we only want to load when serving a train_all-style checkpoint.
        from train_all import OncoSeg as InlineOncoSeg

        model = InlineOncoSeg(
            in_channels=4, num_classes=3, embed_dim=48,
            depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
            dropout_rate=0.1, deep_supervision=False, use_cross_attention=True,
        )
    elif model_source == "src":
        from src.models.oncoseg import OncoSeg

        model = OncoSeg(
            in_channels=4, num_classes=3, embed_dim=48,
            depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
            deep_supervision=False,
        )
    else:
        raise ValueError(f"Unknown model_source: {model_source!r}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict, strict=False)

    predictor = Predictor(
        model=model,
        device=device,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        mc_samples=mc_samples,
    )
    return predictor, f"OncoSeg({model_source})"
