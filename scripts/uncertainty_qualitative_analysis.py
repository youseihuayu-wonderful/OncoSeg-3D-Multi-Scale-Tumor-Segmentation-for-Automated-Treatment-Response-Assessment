"""Uncertainty, qualitative, and failure-mode analysis for OncoSeg.

Outputs (under figures/ and experiments/local_results/):
    - figures/qualitative_comparison.png       — best/median/worst case panel
    - figures/uncertainty_map.png              — MC Dropout entropy on median case
    - figures/uncertainty_calibration.png      — reliability diagram + ECE
    - figures/uncertainty_vs_error.png         — bin uncertainty vs misclass rate
    - experiments/local_results/uncertainty_metrics.json
    - experiments/local_results/failure_analysis.json
    - experiments/local_results/predictions/<subject>/{oncoseg,unet3d,gt,uncertainty}.nii.gz
        (consumed by the RECIST demo notebook)

Run from project root:
    python scripts/uncertainty_qualitative_analysis.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

from src.data.msd_dataset import MSDBrainTumorDataset
from src.data.msd_transforms import get_msd_val_transforms
# IMPORTANT: trained checkpoints come from train_all.py's inline OncoSeg
# (MONAI SwinTransformer encoder + simpler decoder), not src/models/oncoseg.py.
from train_all import OncoSeg

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("uq_analysis")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "Task01_BrainTumour"
RESULTS_DIR = ROOT / "experiments" / "local_results"
FIG_DIR = ROOT / "figures"
PRED_DIR = RESULTS_DIR / "predictions"
ROI = (96, 96, 96)
MC_SAMPLES = 5
REGION_NAMES = ["TC", "WT", "ET"]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_oncoseg(device: torch.device) -> OncoSeg:
    model = OncoSeg(
        in_channels=4,
        num_classes=3,
        embed_dim=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        deep_supervision=True,
        dropout_rate=0.1,
        use_cross_attention=True,
    )
    ck = torch.load(RESULTS_DIR / "oncoseg_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    return model


def load_unet3d(device: torch.device) -> UNet:
    model = UNet(
        spatial_dims=3, in_channels=4, out_channels=3,
        channels=(32, 64, 128, 256), strides=(2, 2, 2),
        num_res_units=2, norm="instance",
    )
    ck = torch.load(RESULTS_DIR / "unet3d_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    return model


def oncoseg_forward(model: OncoSeg):
    return lambda x: model(x)["pred"]


def predict_deterministic(model, image: torch.Tensor, device, kind: str) -> np.ndarray:
    """Sliding-window probability prediction. Returns [3, H, W, D] np.float32."""
    fwd = oncoseg_forward(model) if kind == "oncoseg" else (lambda x: model(x))
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image.to(device), roi_size=ROI, sw_batch_size=1,
            predictor=fwd, overlap=0.5,
        )
    return torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32)


def predict_mc(model: OncoSeg, image: torch.Tensor, device, n_samples: int) -> np.ndarray:
    """Run MC Dropout. Returns stacked probs [n, 3, H, W, D]."""
    samples = []
    # Activate dropout layers
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout3d)):
            m.train()
    fwd = oncoseg_forward(model)
    for i in range(n_samples):
        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=image.to(device), roi_size=ROI, sw_batch_size=1,
                predictor=fwd, overlap=0.5,
            )
        samples.append(torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32))
        log.info(f"  MC sample {i + 1}/{n_samples}")
    model.eval()
    return np.stack(samples, axis=0)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """ECE over flattened binary probs/labels."""
    probs = probs.flatten()
    labels = labels.flatten().astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = probs.size
    bin_data = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if mask.sum() == 0:
            bin_data.append({"lo": float(lo), "hi": float(hi), "count": 0,
                             "conf": 0.0, "acc": 0.0})
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        ece += (mask.sum() / n) * abs(conf - acc)
        bin_data.append({"lo": float(lo), "hi": float(hi), "count": int(mask.sum()),
                         "conf": conf, "acc": acc})
    return float(ece), bin_data


def predictive_entropy(mean_probs: np.ndarray) -> np.ndarray:
    """Per-voxel binary entropy averaged over channels. mean_probs: [3, H, W, D]."""
    eps = 1e-6
    p = np.clip(mean_probs.astype(np.float64), eps, 1.0 - eps)
    ent = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return ent.mean(axis=0).astype(np.float32)  # [H, W, D]


def best_axial_slice(label_3ch: np.ndarray) -> int:
    """Slice index along D with largest WT area (label_3ch: [3, H, W, D])."""
    wt = label_3ch[1]  # WT channel
    areas = wt.sum(axis=(0, 1))
    return int(np.argmax(areas)) if areas.max() > 0 else label_3ch.shape[-1] // 2


def regions_to_rgb(seg_3ch: np.ndarray) -> np.ndarray:
    """Render TC/WT/ET as an RGB overlay on a 2D slice. seg_3ch: [3, H, W] binary."""
    h, w = seg_3ch.shape[1:]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 1] = seg_3ch[1]  # WT → green
    rgb[..., 2] = seg_3ch[0]  # TC → blue
    rgb[..., 0] = seg_3ch[2]  # ET → red
    return np.clip(rgb, 0, 1)


def save_nifti(arr: np.ndarray, path: Path, affine: np.ndarray | None = None):
    if affine is None:
        affine = np.eye(4)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), str(path))


def pick_cases(per_subject: np.ndarray) -> dict[str, int]:
    """Pick best/median/worst subject indices by mean Dice across regions.

    Ignores subjects with NaN Dice (empty GT region → undefined Dice).
    """
    means = np.nanmean(per_subject, axis=1)
    # Exclude subjects with any NaN region score (empty GT) or all-zero (no tumor)
    valid = ~np.isnan(per_subject).any(axis=1) & (means > 0.01)
    valid_idx = np.where(valid)[0]
    sorted_idx = valid_idx[np.argsort(means[valid_idx])]
    return {
        "worst": int(sorted_idx[0]),
        "median": int(sorted_idx[len(sorted_idx) // 2]),
        "best": int(sorted_idx[-1]),
    }


def failure_analysis(per_subject_oncoseg: np.ndarray,
                     per_subject_unet: np.ndarray,
                     subject_names: list[str]) -> dict:
    """Bottom-5 by OncoSeg mean Dice with per-region breakdown."""
    means = per_subject_oncoseg.mean(axis=1)
    bottom = np.argsort(means)[:5]
    cases = []
    for idx in bottom:
        cases.append({
            "subject": subject_names[idx],
            "oncoseg": {r: float(per_subject_oncoseg[idx, i]) for i, r in enumerate(REGION_NAMES)},
            "oncoseg_mean": float(means[idx]),
            "unet3d": {r: float(per_subject_unet[idx, i]) for i, r in enumerate(REGION_NAMES)},
            "unet3d_mean": float(per_subject_unet[idx].mean()),
        })
    # Aggregate failure modes
    bottom_per_region = per_subject_oncoseg[bottom].mean(axis=0)
    overall_per_region = per_subject_oncoseg.mean(axis=0)
    region_drop = {
        REGION_NAMES[i]: {
            "bottom5_mean": float(bottom_per_region[i]),
            "overall_mean": float(overall_per_region[i]),
            "drop": float(overall_per_region[i] - bottom_per_region[i]),
        }
        for i in range(3)
    }
    # Classify dominant failure: which region drops most relative to its overall mean
    rel_drops = {r: region_drop[r]["drop"] / max(region_drop[r]["overall_mean"], 1e-6)
                 for r in REGION_NAMES}
    dominant = max(rel_drops, key=rel_drops.get)
    return {
        "bottom_5_cases": cases,
        "region_breakdown": region_drop,
        "dominant_failure_region": dominant,
        "interpretation": (
            f"Bottom-5 OncoSeg cases drop most heavily on {dominant} "
            f"(relative drop {rel_drops[dominant]:.1%}), suggesting the model "
            f"struggles primarily with this region in hard cases."
        ),
    }


def build_qualitative_figure(cases_data: dict, save_path: Path):
    """3-row x 4-col panel: rows=worst/median/best, cols=image/GT/OncoSeg/UNet3D."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    row_titles = ["Worst", "Median", "Best"]
    col_titles = ["FLAIR", "Ground Truth", "OncoSeg", "UNet3D"]
    for r, key in enumerate(["worst", "median", "best"]):
        d = cases_data[key]
        sl = d["slice_idx"]
        flair = d["image"][0, :, :, sl]  # FLAIR channel
        gt = d["label"][:, :, :, sl]
        onc = d["oncoseg_seg"][:, :, :, sl]
        unet = d["unet3d_seg"][:, :, :, sl]
        # FLAIR background
        axes[r, 0].imshow(flair.T, cmap="gray", origin="lower")
        axes[r, 1].imshow(flair.T, cmap="gray", origin="lower")
        axes[r, 1].imshow(np.transpose(regions_to_rgb(gt), (1, 0, 2)), origin="lower", alpha=0.55)
        axes[r, 2].imshow(flair.T, cmap="gray", origin="lower")
        axes[r, 2].imshow(np.transpose(regions_to_rgb(onc), (1, 0, 2)), origin="lower", alpha=0.55)
        axes[r, 3].imshow(flair.T, cmap="gray", origin="lower")
        axes[r, 3].imshow(np.transpose(regions_to_rgb(unet), (1, 0, 2)), origin="lower", alpha=0.55)
        axes[r, 0].set_ylabel(
            f"{row_titles[r]}\n{d['subject']}\nOncoSeg Dice={d['oncoseg_dice_mean']:.3f}",
            fontsize=10,
        )
        for c in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=12)
    fig.suptitle(
        "Qualitative comparison — RGB overlay: red=ET, green=WT, blue=TC",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def build_uncertainty_figure(case_data: dict, ent_map: np.ndarray, save_path: Path):
    sl = case_data["slice_idx"]
    flair = case_data["image"][0, :, :, sl]
    gt = case_data["label"][:, :, :, sl]
    pred = case_data["oncoseg_seg"][:, :, :, sl]
    ent = ent_map[:, :, sl]
    # Error map: any disagreement across the 3 channels
    err = (pred != gt).any(axis=0).astype(np.float32)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(flair.T, cmap="gray", origin="lower"); axes[0].set_title("FLAIR")
    axes[1].imshow(flair.T, cmap="gray", origin="lower")
    axes[1].imshow(np.transpose(regions_to_rgb(gt), (1, 0, 2)), origin="lower", alpha=0.55)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(flair.T, cmap="gray", origin="lower")
    im = axes[2].imshow(ent.T, cmap="hot", origin="lower", alpha=0.7,
                        vmin=0, vmax=ent.max() if ent.max() > 0 else 1)
    axes[2].set_title(f"MC Dropout Uncertainty\n({MC_SAMPLES} samples)")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    axes[3].imshow(err.T, cmap="Reds", origin="lower")
    axes[3].set_title("Prediction Error")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Uncertainty quantification — {case_data['subject']}", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def build_calibration_figure(bin_data: list, ece: float, save_path: Path):
    confs = [b["conf"] for b in bin_data]
    accs = [b["acc"] for b in bin_data]
    counts = np.array([b["count"] for b in bin_data], dtype=np.float64)
    weights = counts / max(counts.sum(), 1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.bar(np.linspace(0, 1, len(bin_data)), accs, width=1 / len(bin_data),
           edgecolor="black", alpha=0.6, label="Empirical accuracy")
    ax.scatter(confs, accs, s=120 * np.sqrt(weights + 1e-3), color="crimson",
               zorder=5, label="Bin avg (size = #voxels)")
    ax.set_xlabel("Predicted probability (confidence)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability diagram — ECE = {ece:.4f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def build_uncertainty_vs_error(ent_map: np.ndarray, error_3d: np.ndarray, save_path: Path):
    """Bin voxels by entropy, plot mean error rate per bin."""
    ent = ent_map.flatten()
    err = error_3d.flatten().astype(np.float32)
    n_bins = 12
    edges = np.linspace(0, max(ent.max(), 1e-3), n_bins + 1)
    centers, error_rates, sizes = [], [], []
    for i in range(n_bins):
        m = (ent >= edges[i]) & (ent < edges[i + 1] if i < n_bins - 1 else ent <= edges[i + 1])
        if m.sum() == 0:
            continue
        centers.append((edges[i] + edges[i + 1]) / 2)
        error_rates.append(float(err[m].mean()))
        sizes.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(centers, error_rates, c=sizes, s=80, cmap="viridis",
                    edgecolor="black")
    ax.plot(centers, error_rates, color="gray", alpha=0.5)
    ax.set_xlabel("Predictive entropy (binned)")
    ax.set_ylabel("Voxel-wise error rate")
    ax.set_title("Uncertainty vs error — calibration of MC Dropout")
    plt.colorbar(sc, ax=ax, label="# voxels in bin")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    device = get_device()
    log.info(f"Device: {device}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # Load per-subject Dice + dataset
    onc_dice = np.load(RESULTS_DIR / "oncoseg_per_subject_dice.npy")
    unet_dice = np.load(RESULTS_DIR / "unet3d_per_subject_dice.npy")
    log.info(f"Per-subject Dice arrays: OncoSeg {onc_dice.shape}, UNet {unet_dice.shape}")

    transforms = get_msd_val_transforms()
    val_ds_meta = MSDBrainTumorDataset(root_dir=DATA_DIR, split="val", val_split=0.2, seed=42)
    subject_names = [Path(d["image"]).name.replace(".nii.gz", "")
                     for d in val_ds_meta.data_dicts]
    log.info(f"Val subjects: {len(subject_names)}")

    # === Failure analysis (no inference needed) ===
    log.info("Running failure analysis on per-subject Dice arrays")
    fa = failure_analysis(onc_dice, unet_dice, subject_names)
    (RESULTS_DIR / "failure_analysis.json").write_text(json.dumps(fa, indent=2))
    log.info(f"  bottom-5 cases: {[c['subject'] for c in fa['bottom_5_cases']]}")
    log.info(f"  dominant failure region: {fa['dominant_failure_region']}")

    # === Pick best/median/worst cases ===
    case_idx = pick_cases(onc_dice)
    log.info(f"Cases (by OncoSeg mean Dice): {case_idx}")

    # === Build val data only for the picked cases ===
    from monai.data import Dataset

    pick_dicts = [val_ds_meta.data_dicts[i] for i in case_idx.values()]
    case_keys = list(case_idx.keys())
    ds = Dataset(data=pick_dicts, transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # === Load models ===
    log.info("Loading models")
    oncoseg = load_oncoseg(device)
    unet3d = load_unet3d(device)

    cases_data: dict[str, dict] = {}
    median_probs_mc = None
    median_label = None

    for key, batch in zip(case_keys, loader):
        idx = case_idx[key]
        subject = subject_names[idx]
        log.info(f"=== {key.upper()} case: {subject} (Dice={onc_dice[idx].mean():.3f}) ===")
        image = batch["image"]  # [1,4,H,W,D]
        label = batch["label"].squeeze(0).cpu().numpy().astype(np.uint8)  # [3,H,W,D]
        affine = batch["image"].meta["affine"][0].cpu().numpy() \
            if hasattr(batch["image"], "meta") else np.eye(4)

        log.info("  OncoSeg deterministic forward")
        onc_probs = predict_deterministic(oncoseg, image, device, "oncoseg")
        onc_seg = (onc_probs > 0.5).astype(np.uint8)

        log.info("  UNet3D forward")
        unet_probs = predict_deterministic(unet3d, image, device, "unet3d")
        unet_seg = (unet_probs > 0.5).astype(np.uint8)

        sl = best_axial_slice(label)
        cases_data[key] = {
            "subject": subject,
            "image": image.squeeze(0).cpu().numpy(),
            "label": label,
            "oncoseg_seg": onc_seg,
            "unet3d_seg": unet_seg,
            "slice_idx": sl,
            "oncoseg_dice_mean": float(onc_dice[idx].mean()),
            "unet3d_dice_mean": float(unet_dice[idx].mean()),
        }

        # Save predictions for the RECIST demo
        save_nifti(onc_seg.astype(np.float32), PRED_DIR / subject / "oncoseg_seg.nii.gz", affine)
        save_nifti(unet_seg.astype(np.float32), PRED_DIR / subject / "unet3d_seg.nii.gz", affine)
        save_nifti(label.astype(np.float32), PRED_DIR / subject / "gt.nii.gz", affine)

        # Run MC Dropout only on the median case (cost control)
        if key == "median":
            log.info(f"  Running MC Dropout x{MC_SAMPLES} on median case")
            mc_samples = predict_mc(oncoseg, image, device, MC_SAMPLES)
            median_probs_mc = mc_samples
            median_label = label
            mean_probs = mc_samples.mean(axis=0)
            ent = predictive_entropy(mean_probs)
            save_nifti(ent, PRED_DIR / subject / "uncertainty.nii.gz", affine)
            cases_data[key]["uncertainty"] = ent
            cases_data[key]["mc_mean_probs"] = mean_probs

    # === Qualitative figure ===
    log.info("Building qualitative figure")
    build_qualitative_figure(cases_data, FIG_DIR / "qualitative_comparison.png")

    # === Uncertainty figures ===
    log.info("Building uncertainty figures")
    median = cases_data["median"]
    ent_map = median["uncertainty"]
    build_uncertainty_figure(median, ent_map, FIG_DIR / "uncertainty_map.png")

    # ECE on median case (use mean MC probs vs binary label)
    ece, bin_data = expected_calibration_error(median["mc_mean_probs"], median_label, n_bins=15)
    log.info(f"  ECE (median case) = {ece:.4f}")
    build_calibration_figure(bin_data, ece, FIG_DIR / "uncertainty_calibration.png")

    # Uncertainty vs error
    pred = median["oncoseg_seg"]
    error_3d = (pred != median_label).any(axis=0).astype(np.float32)
    build_uncertainty_vs_error(ent_map, error_3d, FIG_DIR / "uncertainty_vs_error.png")

    # === Save metrics ===
    metrics = {
        "mc_samples": MC_SAMPLES,
        "median_case": median["subject"],
        "ece_median_case": ece,
        "mean_entropy_median_case": float(ent_map.mean()),
        "max_entropy_median_case": float(ent_map.max()),
        "mc_variance_mean": float(median_probs_mc.var(axis=0).mean()),
        "calibration_bins": bin_data,
        "cases_used": {k: v["subject"] for k, v in cases_data.items()},
    }
    (RESULTS_DIR / "uncertainty_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(f"Saved uncertainty metrics → {RESULTS_DIR / 'uncertainty_metrics.json'}")
    log.info("Done.")


if __name__ == "__main__":
    main()
