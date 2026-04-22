"""Evaluate saved OncoSeg checkpoint on validation set.

Loads oncoseg_best.pth and computes per-region Dice scores (TC, WT, ET).
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "Task01_BrainTumour"
CHECKPOINT = PROJECT_ROOT / "experiments" / "local_results" / "oncoseg_best.pth"


class ConvertMSDToMultiChanneld(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if isinstance(img, torch.Tensor):
                if img.ndim == 4 and img.shape[0] == 1:
                    img = img.squeeze(0)
                tc = ((img == 2) | (img == 3)).float()
                wt = ((img == 1) | (img == 2) | (img == 3)).float()
                et = (img == 3).float()
                d[key] = torch.stack([tc, wt, et], dim=0)
            else:
                if img.ndim == 4 and img.shape[0] == 1:
                    img = img.squeeze(0)
                tc = ((img == 2) | (img == 3)).astype(np.float32)
                wt = ((img == 1) | (img == 2) | (img == 3)).astype(np.float32)
                et = (img == 3).astype(np.float32)
                d[key] = np.stack([tc, wt, et], axis=0)
        return d


# Import OncoSeg from train_all.py
import importlib.util  # noqa: E402  -- must come after PROJECT_ROOT setup above

spec = importlib.util.spec_from_file_location("train_all", PROJECT_ROOT / "train_all.py")
train_all = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_all)
OncoSeg = train_all.OncoSeg


def main():
    if not CHECKPOINT.exists():
        logger.error(f"Checkpoint not found: {CHECKPOINT}")
        return

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    logger.info(f"Checkpoint from epoch {ckpt['epoch']}, best Dice: {ckpt['best_dice']:.4f}")

    # Build model
    model = OncoSeg(
        in_channels=4, num_classes=3, embed_dim=24,
        depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
        deep_supervision=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # Validation data (same split as training)
    import random as _random
    with open(DATASET_DIR / "dataset.json") as f:
        meta = json.load(f)

    all_data = []
    for entry in meta["training"]:
        img_path = DATASET_DIR / entry["image"]
        lbl_path = DATASET_DIR / entry["label"]
        if img_path.exists() and lbl_path.exists():
            all_data.append({"image": str(img_path), "label": str(lbl_path)})

    rng = _random.Random(42)
    indices = list(range(len(all_data)))
    rng.shuffle(indices)
    n_val = int(len(all_data) * 0.2)
    val_data = [all_data[i] for i in indices[:n_val]]
    logger.info(f"Validation subjects: {len(val_data)}")

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertMSDToMultiChanneld(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_ds = Dataset(data=val_data, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Evaluate
    roi_size = (96, 96, 96)
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")
    per_subject_dice = DiceMetric(include_background=True, reduction="none")

    all_subject_scores = []
    hd95_errors = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            preds = sliding_window_inference(
                inputs=images, roi_size=roi_size, sw_batch_size=1,
                predictor=lambda x: model(x)["pred"], overlap=0.25,
            )
            preds_binary = (torch.sigmoid(preds) > 0.5).float().cpu()
            labels_cpu = labels.cpu()

            dice_metric(y_pred=preds_binary, y=labels_cpu)

            try:
                hd95_metric(y_pred=preds_binary, y=labels_cpu)
            except Exception:
                hd95_errors += 1

            per_subject_dice.reset()
            per_subject_dice(y_pred=preds_binary, y=labels_cpu)
            subj_scores = per_subject_dice.aggregate().numpy()[0]
            all_subject_scores.append(subj_scores)

            if (i + 1) % 10 == 0:
                logger.info(f"  Evaluated {i + 1}/{len(val_loader)} subjects")

    # Aggregate
    scores = dice_metric.aggregate()
    dice_tc = scores[0].item()
    dice_wt = scores[1].item()
    dice_et = scores[2].item()
    dice_mean = scores.mean().item()

    all_subject_scores = np.array(all_subject_scores)
    std_tc = np.std(all_subject_scores[:, 0])
    std_wt = np.std(all_subject_scores[:, 1])
    std_et = np.std(all_subject_scores[:, 2])

    # HD95
    try:
        hd95_scores = hd95_metric.aggregate()
        hd95_tc = float(hd95_scores[0].item())
        hd95_wt = float(hd95_scores[1].item())
        hd95_et = float(hd95_scores[2].item())
        hd95_mean = float(hd95_scores.mean().item())
    except Exception:
        hd95_tc = hd95_wt = hd95_et = hd95_mean = -1.0

    logger.info(f"\n{'='*60}")
    logger.info("ONCOSEG EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Dice TC:   {dice_tc:.4f} +/- {std_tc:.4f}")
    logger.info(f"Dice WT:   {dice_wt:.4f} +/- {std_wt:.4f}")
    logger.info(f"Dice ET:   {dice_et:.4f} +/- {std_et:.4f}")
    logger.info(f"Dice Mean: {dice_mean:.4f}")
    logger.info(f"HD95 TC:   {hd95_tc:.2f} mm")
    logger.info(f"HD95 WT:   {hd95_wt:.2f} mm")
    logger.info(f"HD95 ET:   {hd95_et:.2f} mm")
    logger.info(f"HD95 Mean: {hd95_mean:.2f} mm")
    if hd95_errors > 0:
        logger.info(f"HD95 errors (empty pred/label): {hd95_errors}")
    logger.info(f"{'='*60}")

    # Save results
    results = {
        "model": "oncoseg",
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_best_dice": float(ckpt["best_dice"]),
        "eval_dice_tc": round(float(dice_tc), 4),
        "eval_dice_wt": round(float(dice_wt), 4),
        "eval_dice_et": round(float(dice_et), 4),
        "eval_dice_mean": round(float(dice_mean), 4),
        "eval_std_tc": round(float(std_tc), 4),
        "eval_std_wt": round(float(std_wt), 4),
        "eval_std_et": round(float(std_et), 4),
        "eval_hd95_tc": round(hd95_tc, 2),
        "eval_hd95_wt": round(hd95_wt, 2),
        "eval_hd95_et": round(hd95_et, 2),
        "eval_hd95_mean": round(hd95_mean, 2),
        "num_val_subjects": len(val_data),
        "roi_size": list(roi_size),
        "embed_dim": 24,
    }

    out_path = PROJECT_ROOT / "experiments" / "local_results" / "oncoseg_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
