"""Training loop for OncoSeg and baseline models."""

import logging
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.training.losses import DeepSupervisionLoss, DiceCELoss

logger = logging.getLogger(__name__)


class Trainer:
    """Training and validation loop with experiment tracking."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        # Device — CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

        # Loss
        self.base_loss = DiceCELoss(
            dice_weight=cfg.training.dice_weight,
            ce_weight=cfg.training.ce_weight,
        )
        self.ds_loss = DeepSupervisionLoss(self.base_loss)

        # Optimizer & Scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.max_epochs,
            eta_min=cfg.training.min_lr,
        )

        # Metrics
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean_batch"
        )

        # Tracking
        self.best_dice = 0.0
        self.save_dir = Path(cfg.training.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # W&B
        if WANDB_AVAILABLE and cfg.training.get("use_wandb", False):
            wandb.init(project="oncoseg", config=dict(cfg))

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        epoch_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Support both BraTS (separate modality keys) and MSD (single "image" key)
            if "image" in batch:
                images = batch["image"].to(self.device)
            else:
                images = torch.cat(
                    [batch[k] for k in ["t1n", "t1c", "t2w", "t2f"]], dim=1
                ).to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Main loss
            loss = self.base_loss(outputs["pred"], labels)

            # Deep supervision loss
            if "deep_sup" in outputs:
                loss = loss + self.ds_loss(outputs["deep_sup"], labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        self.scheduler.step()
        avg_loss = epoch_loss / len(self.train_loader)

        if WANDB_AVAILABLE and self.cfg.training.get("use_wandb", False):
            wandb.log({"train/loss": avg_loss, "train/lr": self.scheduler.get_last_lr()[0]})

        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        """Run validation with sliding window inference."""
        self.model.eval()
        self.dice_metric.reset()

        for batch in tqdm(self.val_loader, desc="Validation"):
            if "image" in batch:
                images = batch["image"].to(self.device)
            else:
                images = torch.cat(
                    [batch[k] for k in ["t1n", "t1c", "t2w", "t2f"]], dim=1
                ).to(self.device)
            labels = batch["label"].to(self.device)

            # Sliding window inference for full-resolution validation
            outputs = sliding_window_inference(
                inputs=images,
                roi_size=self.cfg.data.roi_size,
                sw_batch_size=self.cfg.training.sw_batch_size,
                predictor=lambda x: self.model(x)["pred"],
                overlap=0.5,
            )

            preds = torch.softmax(outputs, dim=1)
            preds_binary = (preds > 0.5).float()

            self.dice_metric(y_pred=preds_binary, y=labels)

        dice_scores = self.dice_metric.aggregate()
        metrics = {
            "val/dice_et": dice_scores[0].item(),
            "val/dice_tc": dice_scores[1].item(),
            "val/dice_wt": dice_scores[2].item(),
            "val/dice_mean": dice_scores.mean().item(),
        }

        if WANDB_AVAILABLE and self.cfg.training.get("use_wandb", False):
            wandb.log(metrics)

        # Save best model
        if metrics["val/dice_mean"] > self.best_dice:
            self.best_dice = metrics["val/dice_mean"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_dice": self.best_dice,
                },
                self.save_dir / "best.pth",
            )
            logger.info(f"New best model saved (Dice: {self.best_dice:.4f})")

        return metrics

    def fit(self):
        """Full training loop."""
        logger.info(f"Training on {self.device} for {self.cfg.training.max_epochs} epochs")

        for epoch in range(1, self.cfg.training.max_epochs + 1):
            train_loss = self.train_epoch(epoch)

            if epoch % self.cfg.training.val_interval == 0:
                metrics = self.validate(epoch)
                logger.info(
                    f"Epoch {epoch} | Loss: {train_loss:.4f} | "
                    f"Dice ET: {metrics['val/dice_et']:.4f} | "
                    f"Dice TC: {metrics['val/dice_tc']:.4f} | "
                    f"Dice WT: {metrics['val/dice_wt']:.4f}"
                )

            # Save latest checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                self.save_dir / "latest.pth",
            )

        logger.info(f"Training complete. Best Dice: {self.best_dice:.4f}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Entry point for training."""
    from src.models import OncoSeg

    num_workers = cfg.data.get("num_workers", 4)

    # Data — select loader based on dataset name
    if cfg.data.name == "msd_brain":
        from src.data.msd_dataset import MSDBrainTumorDataset
        from src.data.msd_transforms import get_msd_train_transforms, get_msd_val_transforms

        train_ds = MSDBrainTumorDataset(
            root_dir=cfg.data.root_dir,
            split="train",
            transform=get_msd_train_transforms(roi_size=tuple(cfg.data.roi_size)),
            cache_rate=cfg.data.get("cache_rate", 0.0),
        ).get_dataset()

        val_ds = MSDBrainTumorDataset(
            root_dir=cfg.data.root_dir,
            split="val",
            transform=get_msd_val_transforms(),
            cache_rate=0.0,
        ).get_dataset()
    else:
        from src.data import BraTSDataset, get_train_transforms, get_val_transforms

        train_ds = BraTSDataset(
            root_dir=cfg.data.root_dir,
            split="train",
            transform=get_train_transforms(roi_size=tuple(cfg.data.roi_size)),
        ).get_dataset()

        val_ds = BraTSDataset(
            root_dir=cfg.data.root_dir,
            split="val",
            transform=get_val_transforms(),
        ).get_dataset()

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers
    )

    # Model
    model = OncoSeg(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        embed_dim=cfg.model.embed_dim,
        depths=tuple(cfg.model.depths),
        num_heads=tuple(cfg.model.num_heads),
        deep_supervision=cfg.model.deep_supervision,
    )

    # Train
    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
