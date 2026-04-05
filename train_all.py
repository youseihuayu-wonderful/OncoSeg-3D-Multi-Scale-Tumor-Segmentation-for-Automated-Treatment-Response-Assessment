"""Train all models (OncoSeg + baselines) and generate full results.

Optimized for Apple M1 8GB RAM. Uses MSD Brain Tumor dataset.

Usage:
    python train_all.py                    # Train all models
    python train_all.py --epochs 30        # Fewer epochs for quick test
    python train_all.py --models oncoseg   # Train only OncoSeg
"""

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet, SwinUNETR
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    SpatialPadd,
    Spacingd,
)
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "Task01_BrainTumour"
SAVE_DIR = PROJECT_ROOT / "experiments" / "local_results"


# =====================================================================
# MSD Label Conversion (same as Colab notebook)
# =====================================================================
class ConvertMSDToMultiChanneld(MapTransform):
    """MSD labels {0,1,2,3} -> 3 channels: TC(2+3), WT(1+2+3), ET(3)."""

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


# =====================================================================
# Loss Functions
# =====================================================================
class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss(sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)
        self.ce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.dice_weight * self.dice(pred, target) + self.ce_weight * self.ce(pred, target)


class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, predictions, target):
        n = len(predictions)
        raw = [0.5**i for i in range(1, n + 1)]
        total = sum(raw)
        weights = [w / total for w in raw]
        loss = 0.0
        for pred, w in zip(predictions, weights):
            loss += w * self.base_loss(pred, target)
        return loss


# =====================================================================
# OncoSeg Model (self-contained, no imports from src/)
# =====================================================================
from monai.networks.nets.swin_unetr import SwinTransformer


class CrossAttentionSkip(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = decoder_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(decoder_dim, decoder_dim)
        self.k_proj = nn.Linear(encoder_dim, decoder_dim)
        self.v_proj = nn.Linear(encoder_dim, decoder_dim)
        self.out_proj = nn.Linear(decoder_dim, decoder_dim)
        self.norm_enc = nn.LayerNorm(encoder_dim)
        self.norm_dec = nn.LayerNorm(decoder_dim)
        self.norm_out = nn.LayerNorm(decoder_dim)
        self.ffn = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim * 4),
            nn.GELU(),
            nn.Linear(decoder_dim * 4, decoder_dim),
        )

    def forward(self, encoder_feat, decoder_feat):
        B, C_dec, H, W, D = decoder_feat.shape
        enc_seq = encoder_feat.flatten(2).transpose(1, 2)
        dec_seq = decoder_feat.flatten(2).transpose(1, 2)
        enc_seq = self.norm_enc(enc_seq)
        dec_seq_normed = self.norm_dec(dec_seq)
        Q = self.q_proj(dec_seq_normed).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(enc_seq).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(enc_seq).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, -1, C_dec)
        out = self.out_proj(out)
        out = dec_seq + out
        out = out + self.ffn(self.norm_out(out))
        return out.transpose(1, 2).reshape(B, C_dec, H, W, D)


class OncoSeg(nn.Module):
    def __init__(self, in_channels=4, num_classes=3, embed_dim=48,
                 depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                 window_size=(7, 7, 7), dropout_rate=0.1, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        self.encoder = SwinTransformer(
            in_chans=in_channels, embed_dim=embed_dim, window_size=window_size,
            patch_size=(4, 4, 4), depths=depths, num_heads=num_heads,
            mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm, spatial_dims=3,
        )
        dims = [embed_dim * (2**i) for i in range(len(depths))]

        self.cross_attn_skips = nn.ModuleList([
            CrossAttentionSkip(dims[i], dims[i], num_heads=max(dims[i] // 48, 1))
            for i in range(1, len(dims) - 1)
        ])

        self.decoders = nn.ModuleList()
        reversed_dims = list(reversed(dims))
        for i in range(len(reversed_dims) - 1):
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose3d(reversed_dims[i], reversed_dims[i + 1], kernel_size=2, stride=2),
                nn.InstanceNorm3d(reversed_dims[i + 1]),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(reversed_dims[i + 1], reversed_dims[i + 1], kernel_size=3, padding=1),
                nn.InstanceNorm3d(reversed_dims[i + 1]),
                nn.LeakyReLU(inplace=True),
            ))

        self.final_conv = nn.Sequential(
            nn.ConvTranspose3d(dims[0], dims[0], kernel_size=4, stride=4),
            nn.Conv3d(dims[0], num_classes, kernel_size=1),
        )

        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv3d(d, num_classes, kernel_size=1) for d in reversed_dims[1:]
            ])

        self.mc_dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self, x):
        B, C, H, W, D = x.shape
        enc_features = self.encoder(x)
        # MONAI SwinTransformer returns 5 features: [patch_embed, stage1..4]
        # We use first 4 (matching our dims), skip the 5th (2x deeper)
        stage_features = enc_features[:len(self.decoders) + 1]

        x_dec = self.mc_dropout(stage_features[-1])
        ds_outputs = []

        for i, decoder in enumerate(self.decoders):
            x_dec = decoder(x_dec)
            skip_idx = len(stage_features) - 2 - i
            skip = stage_features[skip_idx]
            if x_dec.shape[2:] != skip.shape[2:]:
                x_dec = F.interpolate(x_dec, size=skip.shape[2:], mode="trilinear", align_corners=False)
            if skip_idx > 0:
                x_dec = self.cross_attn_skips[skip_idx - 1](encoder_feat=skip, decoder_feat=x_dec)
            else:
                x_dec = x_dec + skip
            ds_outputs.append(x_dec)

        pred = self.final_conv(x_dec)
        if pred.shape[2:] != (H, W, D):
            pred = F.interpolate(pred, size=(H, W, D), mode="trilinear", align_corners=False)

        outputs = {"pred": pred}
        if self.deep_supervision and self.training:
            ds_preds = []
            for feat, head in zip(ds_outputs, self.ds_heads):
                ds_pred = head(feat)
                ds_pred = F.interpolate(ds_pred, size=(H, W, D), mode="trilinear", align_corners=False)
                ds_preds.append(ds_pred)
            outputs["deep_sup"] = ds_preds
        return outputs


# =====================================================================
# Model Factory
# =====================================================================
NUM_CLASSES = 3

def build_model(name, roi_size, embed_dim=48):
    if name == "oncoseg":
        return OncoSeg(
            in_channels=4, num_classes=NUM_CLASSES, embed_dim=embed_dim,
            depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), deep_supervision=True,
        )
    elif name == "unet3d":
        return UNet(
            spatial_dims=3, in_channels=4, out_channels=NUM_CLASSES,
            channels=(32, 64, 128, 256), strides=(2, 2, 2),
            num_res_units=2, norm="instance",
        )
    elif name == "swin_unetr":
        return SwinUNETR(
            in_channels=4, out_channels=NUM_CLASSES,
            feature_size=embed_dim, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
            norm_name="instance", spatial_dims=3,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


# =====================================================================
# Data Loading
# =====================================================================
def build_data(roi_size, val_split=0.2, seed=42):
    import random as _random

    with open(DATASET_DIR / "dataset.json") as f:
        meta = json.load(f)

    all_data = []
    for entry in meta["training"]:
        img_path = DATASET_DIR / entry["image"]
        lbl_path = DATASET_DIR / entry["label"]
        if img_path.exists() and lbl_path.exists():
            all_data.append({"image": str(img_path), "label": str(lbl_path)})

    rng = _random.Random(seed)
    indices = list(range(len(all_data)))
    rng.shuffle(indices)
    n_val = int(len(all_data) * val_split)
    val_data = [all_data[i] for i in indices[:n_val]]
    train_data = [all_data[i] for i in indices[n_val:]]

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertMSDToMultiChanneld(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label"]),
    ])

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

    from monai.data import Dataset
    train_ds = Dataset(data=train_data, transform=train_transforms)
    val_ds = Dataset(data=val_data, transform=val_transforms)

    return train_ds, val_ds, len(train_data), len(val_data)


# =====================================================================
# Training
# =====================================================================
def train_model(name, train_loader, val_loader, device, roi_size,
                max_epochs=50, lr=1e-4, val_interval=5, embed_dim=48):
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {name}")
    logger.info(f"{'='*60}")

    model = build_model(name, roi_size, embed_dim).to(device)
    is_oncoseg = name == "oncoseg"
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Parameters: {n_params:.1f}M")

    base_loss = DiceCELoss()
    ds_loss = DeepSupervisionLoss(base_loss) if is_oncoseg else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    history = {"train_loss": [], "val_dice_tc": [], "val_dice_wt": [], "val_dice_et": [],
               "val_dice_mean": [], "best_dice": 0.0, "best_epoch": 0}

    save_path = SAVE_DIR / f"{name}_best.pth"
    best_dice = 0.0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{name}] Epoch {epoch}/{max_epochs}", leave=False)

        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()

            if is_oncoseg:
                outputs = model(images)
                loss = base_loss(outputs["pred"], labels)
                if "deep_sup" in outputs:
                    loss = loss + 0.5 * ds_loss(outputs["deep_sup"], labels)
            else:
                pred = model(images)
                if isinstance(pred, dict):
                    pred = pred["pred"]
                loss = base_loss(pred, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_loss)

        if epoch % val_interval == 0 or epoch == max_epochs:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating", leave=False):
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)

                    if is_oncoseg:
                        pred_fn = lambda x: model(x)["pred"]
                    else:
                        def pred_fn(x):
                            out = model(x)
                            return out["pred"] if isinstance(out, dict) else out

                    preds = sliding_window_inference(
                        inputs=images, roi_size=roi_size, sw_batch_size=1,
                        predictor=pred_fn, overlap=0.25,
                    )
                    preds_binary = (torch.sigmoid(preds) > 0.5).float()
                    dice_metric(y_pred=preds_binary, y=labels)

            scores = dice_metric.aggregate()
            dice_tc, dice_wt, dice_et = scores[0].item(), scores[1].item(), scores[2].item()
            dice_mean = scores.mean().item()

            history["val_dice_tc"].append(dice_tc)
            history["val_dice_wt"].append(dice_wt)
            history["val_dice_et"].append(dice_et)
            history["val_dice_mean"].append(dice_mean)

            logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f} | "
                        f"Dice TC: {dice_tc:.4f} WT: {dice_wt:.4f} ET: {dice_et:.4f} Mean: {dice_mean:.4f}")

            if dice_mean > best_dice:
                best_dice = dice_mean
                history["best_dice"] = best_dice
                history["best_epoch"] = epoch
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                             "best_dice": best_dice}, save_path)
                logger.info(f"  -> New best (Dice: {best_dice:.4f})")
        else:
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

    logger.info(f"Done. Best Dice: {history['best_dice']:.4f} at epoch {history['best_epoch']}")

    del model, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return history


# =====================================================================
# Figures
# =====================================================================
def generate_figures(all_histories, max_epochs, val_interval):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"oncoseg": "#e74c3c", "unet3d": "#3498db", "swin_unetr": "#2ecc71"}

    for name, hist in all_histories.items():
        c = colors.get(name, "#333")
        axes[0].plot(hist["train_loss"], label=name, color=c, linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    val_epochs = list(range(val_interval, max_epochs + 1, val_interval))
    for name, hist in all_histories.items():
        if hist["val_dice_mean"]:
            c = colors.get(name, "#333")
            axes[1].plot(val_epochs[:len(hist["val_dice_mean"])], hist["val_dice_mean"],
                         label=f"{name} (best: {hist['best_dice']:.4f})",
                         color=c, linewidth=2, marker="o", markersize=3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Dice Score")
    axes[1].set_title("Validation Dice Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("OncoSeg — Training on Real MSD Brain Tumor Data", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved training curves to {SAVE_DIR / 'training_curves.png'}")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Train all models locally")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--roi-size", type=int, default=96)
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--models", type=str, nargs="+", default=["oncoseg", "unet3d", "swin_unetr"])
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Seed
    torch.manual_seed(42)
    np.random.seed(42)

    roi_size = (args.roi_size, args.roi_size, args.roi_size)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Data
    logger.info("Loading MSD Brain Tumor dataset...")
    train_ds, val_ds, n_train, n_val = build_data(roi_size)
    logger.info(f"Train: {n_train} | Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Train all models
    all_histories = {}
    start_time = time.time()

    for model_name in args.models:
        all_histories[model_name] = train_model(
            model_name, train_loader, val_loader, device, roi_size,
            max_epochs=args.epochs, lr=args.lr, val_interval=args.val_interval,
            embed_dim=args.embed_dim,
        )

    total_time = time.time() - start_time

    # Generate figures
    generate_figures(all_histories, args.epochs, args.val_interval)

    # Save results
    results = {
        "config": {
            "epochs": args.epochs, "roi_size": args.roi_size, "embed_dim": args.embed_dim,
            "batch_size": args.batch_size, "lr": args.lr, "device": str(device),
            "train_subjects": n_train, "val_subjects": n_val,
            "total_time_minutes": total_time / 60,
        },
        "models": {},
    }
    for name, hist in all_histories.items():
        results["models"][name] = {
            "best_dice": hist["best_dice"],
            "best_epoch": hist["best_epoch"],
            "final_loss": hist["train_loss"][-1] if hist["train_loss"] else None,
        }

    with open(SAVE_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total training time: {total_time/60:.1f} minutes")
    for name, hist in all_histories.items():
        logger.info(f"  {name:15s}: Best Dice = {hist['best_dice']:.4f} (epoch {hist['best_epoch']})")
    logger.info(f"Results saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
