"""Inference pipeline for OncoSeg: predict segmentation masks from NIfTI volumes.

Supports:
    - Single-volume and batch prediction
    - Sliding window inference for arbitrary input sizes
    - MC Dropout uncertainty estimation
    - NIfTI output with original affine/header preserved
    - Optional RECIST measurement on predictions
"""

import json
import logging
from pathlib import Path

import hydra
import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)
from omegaconf import DictConfig
from tqdm import tqdm

from src.models.oncoseg import OncoSeg
from src.response.recist import RECISTMeasurer

logger = logging.getLogger(__name__)


def build_inference_transforms(
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Minimal preprocessing transforms for inference on BraTS-format inputs."""
    keys = ["t1n", "t1c", "t2w", "t2f"]
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear",) * 4),
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
            EnsureTyped(keys=keys),
        ]
    )


def build_single_image_transforms(
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Preprocessing for single 4D NIfTI (MSD-format) inference."""
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=pixdim, mode=("bilinear",)),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )


def discover_subjects(input_dir: Path) -> list[dict]:
    """Scan a BraTS-format directory for subjects and build data dicts."""
    modalities = ["t1n", "t1c", "t2w", "t2f"]
    subjects = []

    for subject_dir in sorted(input_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        entry = {"subject_id": subject_dir.name}
        for mod in modalities:
            matches = list(subject_dir.glob(f"*{mod}.nii.gz"))
            if matches:
                entry[mod] = str(matches[0])

        if all(mod in entry for mod in modalities):
            subjects.append(entry)

    return subjects


class Predictor:
    """Run inference with a trained OncoSeg model."""

    def __init__(
        self,
        model: OncoSeg,
        device: torch.device,
        roi_size: tuple[int, int, int] = (128, 128, 128),
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        mc_samples: int = 0,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mc_samples = mc_samples
        self.measurer = RECISTMeasurer()

    def _forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["pred"]

    @torch.no_grad()
    def predict_volume(self, image: torch.Tensor) -> dict[str, np.ndarray]:
        """Run prediction on a single preprocessed volume.

        Args:
            image: Preprocessed input tensor [1, C, H, W, D] on CPU or GPU.

        Returns:
            Dictionary with:
                - "segmentation": Integer label map [H, W, D]
                - "probabilities": Softmax probabilities [num_classes, H, W, D]
                - "uncertainty": Entropy uncertainty map [H, W, D] (if mc_samples > 0)
        """
        image = image.to(self.device)

        logits = sliding_window_inference(
            inputs=image,
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self._forward_fn,
            overlap=self.overlap,
        )

        probs = torch.softmax(logits, dim=1)
        seg = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        probs_np = probs.squeeze(0).cpu().numpy()

        result = {
            "segmentation": seg,
            "probabilities": probs_np,
        }

        if self.mc_samples > 0:
            uncertainty = self._estimate_uncertainty(image)
            result["uncertainty"] = uncertainty

        return result

    def _estimate_uncertainty(self, image: torch.Tensor) -> np.ndarray:
        """Estimate uncertainty via MC Dropout."""
        self.model.mc_dropout.train()
        predictions = []

        for _ in range(self.mc_samples):
            enc_features = self.model.encoder(image)
            enc_features[-1] = self.model.mc_dropout(enc_features[-1])
            dec_out = self.model.decoder(enc_features, self.model.cross_attn_skips)
            prob = torch.softmax(dec_out["pred"], dim=1)
            predictions.append(prob)

        self.model.mc_dropout.eval()

        stacked = torch.stack(predictions, dim=0)
        mean_pred = stacked.mean(dim=0)
        entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=1).squeeze(0)

        return entropy.cpu().numpy()

    def predict_and_save(
        self,
        data_dict: dict,
        transforms: Compose,
        output_dir: Path,
        save_probabilities: bool = False,
        measure_recist: bool = False,
        pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> dict:
        """Full prediction pipeline: load, preprocess, predict, save NIfTI.

        Args:
            data_dict: Dictionary with file paths (BraTS-format keys or "image" key).
            transforms: Preprocessing transforms.
            output_dir: Where to save outputs.
            save_probabilities: Whether to save softmax probability maps.
            measure_recist: Whether to compute RECIST measurements.
            pixdim: Voxel spacing for RECIST measurement.

        Returns:
            Dictionary with output paths and optional RECIST measurements.
        """
        subject_id = data_dict.get("subject_id", "unknown")
        processed = transforms(data_dict)

        # Stack modalities into a single tensor
        if "image" in processed:
            image = processed["image"].unsqueeze(0)
        else:
            image = torch.cat(
                [processed[k] for k in ["t1n", "t1c", "t2w", "t2f"]], dim=0
            ).unsqueeze(0)

        result = self.predict_volume(image)

        # Load reference NIfTI for affine/header
        if "t1c" in data_dict:
            ref_nii = nib.load(data_dict["t1c"])
        elif "image" in data_dict:
            ref_nii = nib.load(data_dict["image"])
        else:
            ref_nii = None

        affine = ref_nii.affine if ref_nii is not None else np.eye(4)

        subject_dir = output_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        outputs = {"subject_id": subject_id}

        # Save segmentation
        seg_path = subject_dir / "segmentation.nii.gz"
        seg_nii = nib.Nifti1Image(result["segmentation"], affine)
        nib.save(seg_nii, seg_path)
        outputs["segmentation_path"] = str(seg_path)

        # Save probabilities
        if save_probabilities:
            prob_path = subject_dir / "probabilities.nii.gz"
            prob_data = np.transpose(result["probabilities"], (1, 2, 3, 0))
            prob_nii = nib.Nifti1Image(prob_data.astype(np.float32), affine)
            nib.save(prob_nii, prob_path)
            outputs["probabilities_path"] = str(prob_path)

        # Save uncertainty
        if "uncertainty" in result:
            unc_path = subject_dir / "uncertainty.nii.gz"
            unc_nii = nib.Nifti1Image(result["uncertainty"].astype(np.float32), affine)
            nib.save(unc_nii, unc_path)
            outputs["uncertainty_path"] = str(unc_path)

        # RECIST measurements
        if measure_recist:
            # Measure enhancing tumor region (label 3)
            et_mask = (result["segmentation"] == 3).astype(np.uint8)
            lesions = self.measurer.measure_lesions(et_mask, pixdim)
            outputs["recist"] = {
                "num_lesions": len(lesions),
                "sum_longest_diameter_mm": sum(les["longest_diameter_mm"] for les in lesions),
                "total_volume_mm3": sum(les["volume_mm3"] for les in lesions),
                "lesions": lesions,
            }

        return outputs


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Entry point for oncoseg-predict CLI."""
    input_path = Path(cfg.get("input", "data/raw/brats2023/test"))
    output_dir = Path(cfg.get("output", "predictions"))
    checkpoint_path = Path(cfg.get("checkpoint", "experiments/oncoseg/best.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = OncoSeg(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        embed_dim=cfg.model.embed_dim,
        depths=tuple(cfg.model.depths),
        num_heads=tuple(cfg.model.num_heads),
        deep_supervision=False,
    )

    # Load checkpoint
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}, using random weights")

    mc_samples = cfg.evaluation.get("mc_samples", 0)

    predictor = Predictor(
        model=model,
        device=device,
        roi_size=tuple(cfg.data.roi_size),
        sw_batch_size=cfg.training.sw_batch_size,
        mc_samples=mc_samples,
    )

    transforms = build_inference_transforms(pixdim=tuple(cfg.data.pixdim))
    pixdim = tuple(cfg.data.pixdim)

    # Discover subjects
    subjects = discover_subjects(input_path)
    logger.info(f"Found {len(subjects)} subjects in {input_path}")

    all_outputs = []
    for subject in tqdm(subjects, desc="Predicting"):
        result = predictor.predict_and_save(
            data_dict=subject,
            transforms=transforms,
            output_dir=output_dir,
            save_probabilities=cfg.get("save_probabilities", False),
            measure_recist=cfg.get("measure_recist", False),
            pixdim=pixdim,
        )
        all_outputs.append(result)
        logger.info(f"  {result['subject_id']}: saved to {result['segmentation_path']}")

    # Save summary
    summary_path = output_dir / "predictions_summary.json"
    summary = {
        "num_subjects": len(all_outputs),
        "checkpoint": str(checkpoint_path),
        "mc_samples": mc_samples,
        "subjects": all_outputs,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Predictions complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
