"""Local training script for OncoSeg on Apple M1/M2 (MPS) or CPU.

Usage:
    # First download the dataset (only needed once, ~7.1 GB):
    python train_local.py --download-only

    # Then train:
    python train_local.py

    # Train with custom epochs:
    python train_local.py --epochs 50

    # Use CPU instead of MPS:
    python train_local.py --device cpu
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from tqdm import tqdm

from src.data.msd_dataset import MSDBrainTumorDataset
from src.data.msd_transforms import get_msd_train_transforms, get_msd_val_transforms
from src.models.oncoseg import OncoSeg
from src.training.losses import DeepSupervisionLoss, DiceCELoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# === Constants for M1 8GB ===
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_DIR = DATA_DIR / "Task01_BrainTumour"
SAVE_DIR = PROJECT_ROOT / "experiments" / "oncoseg_msd"
MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"


def download_msd_dataset():
    """Download and extract MSD Brain Tumor dataset (~7.1 GB)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_DIR.exists():
        meta_path = DATASET_DIR / "dataset.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            n_images = len(list((DATASET_DIR / "imagesTr").glob("*.nii.gz")))
            logger.info(
                f"Dataset already exists: {meta['name']} — "
                f"{n_images}/{meta['numTraining']} training images"
            )
            if n_images == meta["numTraining"]:
                return
            logger.warning("File count mismatch — re-downloading")

    tar_path = DATA_DIR / "Task01_BrainTumour.tar"

    logger.info(f"Downloading MSD Task01_BrainTumour (~7.1 GB) from {MSD_URL}")
    logger.info("This will take a while depending on your internet connection...")

    # Use curl (available on macOS by default) instead of wget
    subprocess.run(
        ["curl", "-L", "--progress-bar", "-o", str(tar_path), MSD_URL],
        check=True,
    )

    logger.info("Extracting archive...")
    subprocess.run(["tar", "-xf", str(tar_path), "-C", str(DATA_DIR)], check=True)

    # Clean up tar file to save disk space
    tar_path.unlink()
    logger.info(f"Dataset ready at {DATASET_DIR}")

    # Verify
    n_images = len(list((DATASET_DIR / "imagesTr").glob("*.nii.gz")))
    n_labels = len(list((DATASET_DIR / "labelsTr").glob("*.nii.gz")))
    logger.info(f"Found {n_images} training images, {n_labels} training labels")


def get_device(requested: str) -> torch.device:
    """Select best available device."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    base_loss: DiceCELoss,
    ds_loss: DeepSupervisionLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Run one training epoch."""
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = base_loss(outputs["pred"], labels)
        if "deep_sup" in outputs:
            loss = loss + ds_loss(outputs["deep_sup"], labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_loss / max(len(loader), 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    roi_size: tuple[int, int, int],
) -> dict[str, float]:
    """Run validation with sliding window inference."""
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    for batch in tqdm(loader, desc="Validation"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = sliding_window_inference(
            inputs=images,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=lambda x: model(x)["pred"],
            overlap=0.25,
        )

        preds = torch.softmax(outputs, dim=1)
        preds_binary = (preds > 0.5).float()
        dice_metric(y_pred=preds_binary, y=labels)

    dice_scores = dice_metric.aggregate()
    return {
        "dice_wt": dice_scores[0].item(),
        "dice_tc": dice_scores[1].item(),
        "dice_et": dice_scores[2].item(),
        "dice_mean": dice_scores.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train OncoSeg locally")
    parser.add_argument("--download-only", action="store_true", help="Only download dataset, don't train")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (1 recommended for 8GB RAM)")
    parser.add_argument("--roi-size", type=int, default=96, help="ROI crop size (96 saves memory vs 128)")
    parser.add_argument("--embed-dim", type=int, default=24, help="Model embed dim (24=small for M1, 48=full)")
    parser.add_argument("--val-interval", type=int, default=5, help="Validate every N epochs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    # Step 1: Download data
    download_msd_dataset()
    if args.download_only:
        logger.info("Download complete. Run without --download-only to start training.")
        return

    # Step 2: Device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    roi_size = (args.roi_size, args.roi_size, args.roi_size)

    # Step 3: Data
    logger.info("Loading MSD Brain Tumor dataset...")
    train_ds = MSDBrainTumorDataset(
        root_dir=str(DATASET_DIR),
        split="train",
        transform=get_msd_train_transforms(roi_size=roi_size),
        cache_rate=0.0,  # No caching to save RAM
        val_split=args.val_split,
    ).get_dataset()

    val_ds = MSDBrainTumorDataset(
        root_dir=str(DATASET_DIR),
        split="val",
        transform=get_msd_val_transforms(),
        cache_rate=0.0,
        val_split=args.val_split,
    ).get_dataset()

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    logger.info(f"Train: {len(train_ds)} subjects | Val: {len(val_ds)} subjects")

    # Step 4: Model
    # Adjust num_heads to match embed_dim
    if args.embed_dim == 24:
        depths = (2, 2, 2, 2)
        num_heads = (3, 6, 12, 24)
    else:
        depths = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)

    model = OncoSeg(
        in_channels=4,
        num_classes=4,
        embed_dim=args.embed_dim,
        depths=depths,
        num_heads=num_heads,
        deep_supervision=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"OncoSeg model: {n_params:.1f}M parameters (embed_dim={args.embed_dim})")

    # Step 5: Loss, optimizer, scheduler
    base_loss = DiceCELoss(dice_weight=0.5, ce_weight=0.5)
    ds_loss = DeepSupervisionLoss(base_loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Step 6: Training loop
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0
    results_log = []

    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"ROI: {roi_size} | Batch: {args.batch_size} | LR: {args.lr}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, base_loss, ds_loss, optimizer, device, epoch
        )
        scheduler.step()

        log_entry = {"epoch": epoch, "train_loss": train_loss, "lr": scheduler.get_last_lr()[0]}

        if epoch % args.val_interval == 0 or epoch == args.epochs:
            metrics = validate(model, val_loader, device, roi_size)
            log_entry.update(metrics)
            logger.info(
                f"Epoch {epoch}/{args.epochs} | Loss: {train_loss:.4f} | "
                f"Dice WT: {metrics['dice_wt']:.4f} | "
                f"TC: {metrics['dice_tc']:.4f} | "
                f"ET: {metrics['dice_et']:.4f} | "
                f"Mean: {metrics['dice_mean']:.4f}"
            )

            if metrics["dice_mean"] > best_dice:
                best_dice = metrics["dice_mean"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_dice": best_dice,
                        "args": vars(args),
                    },
                    SAVE_DIR / "best.pth",
                )
                logger.info(f"  -> New best model saved (Dice: {best_dice:.4f})")
        else:
            logger.info(f"Epoch {epoch}/{args.epochs} | Loss: {train_loss:.4f}")

        results_log.append(log_entry)

        # Save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            SAVE_DIR / "latest.pth",
        )

    # Save training log
    with open(SAVE_DIR / "training_log.json", "w") as f:
        json.dump(results_log, f, indent=2)

    logger.info(f"\nTraining complete! Best mean Dice: {best_dice:.4f}")
    logger.info(f"Checkpoints saved to: {SAVE_DIR}")
    logger.info(f"Training log: {SAVE_DIR / 'training_log.json'}")


if __name__ == "__main__":
    main()
