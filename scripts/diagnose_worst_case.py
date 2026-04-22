"""Diagnose why BRATS_077 is OncoSeg's worst-case subject.

Compares:
    - Overall tumor burden (voxels per region)
    - Tumor size percentile vs rest of val set
    - Intensity contrast between tumor and non-tumor tissue
    - Small-lesion / scattered-lesion structure
    - Spatial location (is the tumor at volume edges, likely cropped?)

Output: experiments/local_results/worst_case_diagnosis.json + a visual
comparison figure showing BRATS_077 vs the median case (BRATS_425).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

from src.data.msd_dataset import MSDBrainTumorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("diagnose")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "Task01_BrainTumour"
RESULTS_DIR = ROOT / "experiments" / "local_results"
FIG_DIR = ROOT / "figures"

# Labels in MSD: 0=bg, 1=edema, 2=non-enhancing, 3=enhancing
REGION_SPEC = {
    "WT": [1, 2, 3],
    "TC": [2, 3],
    "ET": [3],
}


def load_case(image_path: Path, label_path: Path):
    img = nib.load(image_path).get_fdata()  # [H,W,D,4] for MSD
    lbl = nib.load(label_path).get_fdata().astype(np.int16)  # [H,W,D]
    return img, lbl


def region_stats(img: np.ndarray, lbl: np.ndarray) -> dict:
    """Per-region voxel counts + mean/std intensity across the 4 modalities."""
    stats = {}
    total_vox = lbl.size
    for name, labels in REGION_SPEC.items():
        mask = np.isin(lbl, labels)
        n_vox = int(mask.sum())
        fraction = n_vox / total_vox
        # Intensities across 4 modalities inside the mask
        if n_vox > 0:
            mod_means = [float(img[..., m][mask].mean()) for m in range(4)]
            mod_stds = [float(img[..., m][mask].std()) for m in range(4)]
        else:
            mod_means = [0.0] * 4
            mod_stds = [0.0] * 4
        stats[name] = {
            "voxels": n_vox,
            "fraction_of_volume": fraction,
            "modality_means": mod_means,
            "modality_stds": mod_stds,
        }
    return stats


def contrast_ratio(img: np.ndarray, lbl: np.ndarray) -> dict:
    """Mean-absolute-difference between tumor and non-tumor tissue, per modality.

    Only considers the brain (non-zero) region as 'non-tumor' to avoid air voxels.
    """
    out = {}
    wt_mask = np.isin(lbl, REGION_SPEC["WT"])
    brain_mask = img[..., 0] != 0  # T1w (or FLAIR depending on index) non-zero
    non_tumor_mask = brain_mask & ~wt_mask
    for m in range(4):
        vol = img[..., m]
        if wt_mask.sum() == 0 or non_tumor_mask.sum() == 0:
            out[f"mod{m}"] = None
            continue
        tumor_mean = vol[wt_mask].mean()
        bg_mean = vol[non_tumor_mask].mean()
        bg_std = vol[non_tumor_mask].std()
        # Normalised contrast: |Δμ| / σ_bg  (Michelson-like, unit-free)
        c = abs(tumor_mean - bg_mean) / (bg_std + 1e-6)
        out[f"mod{m}"] = float(c)
    return out


def connected_component_profile(lbl: np.ndarray) -> dict:
    """How fragmented is the tumor? (many small lesions are harder)."""
    profile = {}
    for name, labels in REGION_SPEC.items():
        mask = np.isin(lbl, labels)
        if mask.sum() == 0:
            profile[name] = {"num_components": 0, "largest_frac": 0.0}
            continue
        labeled, n = ndimage.label(mask)
        if n == 0:
            profile[name] = {"num_components": 0, "largest_frac": 0.0}
            continue
        sizes = ndimage.sum_labels(mask, labeled, index=range(1, n + 1))
        sizes = np.atleast_1d(sizes)
        profile[name] = {
            "num_components": int(n),
            "largest_frac": float(sizes.max() / sizes.sum()),
            "mean_size": float(sizes.mean()),
        }
    return profile


def volume_center_offset(lbl: np.ndarray) -> dict:
    """How off-center is the tumor? Bias toward edges → cropping risk during training."""
    wt_mask = np.isin(lbl, REGION_SPEC["WT"])
    if wt_mask.sum() == 0:
        return {"center_offset_norm": None}
    com = np.array(ndimage.center_of_mass(wt_mask))
    shape = np.array(lbl.shape)
    center = shape / 2
    offset = np.linalg.norm((com - center) / shape)  # normalised
    return {"center_offset_norm": float(offset), "com": com.tolist()}


def cohort_size_percentile(wt_voxels: int, all_wt_voxels: list[int]) -> float:
    arr = np.sort(all_wt_voxels)
    rank = np.searchsorted(arr, wt_voxels)
    return float(rank / len(arr)) * 100.0


def render_comparison(case_a, case_b, save_path: Path):
    """Side-by-side FLAIR + GT overlay for two cases (worst vs median)."""
    (img_a, lbl_a, name_a), (img_b, lbl_b, name_b) = case_a, case_b

    def best_slice(lbl):
        wt = np.isin(lbl, REGION_SPEC["WT"])
        areas = wt.sum(axis=(0, 1))
        return int(np.argmax(areas)) if areas.max() > 0 else lbl.shape[-1] // 2

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, (img, lbl, name) in enumerate([(img_a, lbl_a, name_a), (img_b, lbl_b, name_b)]):
        sl = best_slice(lbl)
        flair = img[..., 0]  # modality 0
        t1c = img[..., 1]
        label_slice = lbl[:, :, sl]
        axes[row, 0].imshow(flair[:, :, sl].T, cmap="gray", origin="lower")
        axes[row, 0].set_title(f"{name} — FLAIR (slice {sl})")
        axes[row, 1].imshow(t1c[:, :, sl].T, cmap="gray", origin="lower")
        axes[row, 1].set_title(f"{name} — T1 (slice {sl})")
        axes[row, 2].imshow(flair[:, :, sl].T, cmap="gray", origin="lower")
        # Overlay: edema=yellow, non-enh=blue, enh=red
        rgb = np.zeros((*label_slice.shape, 4))
        rgb[..., 0] = (label_slice == 3) * 1.0  # ET = red
        rgb[..., 1] = (label_slice == 1) * 1.0  # edema = green/yellow
        rgb[..., 2] = (label_slice == 2) * 1.0  # non-enh = blue
        rgb[..., 3] = ((label_slice > 0) * 0.5).astype(float)
        axes[row, 2].imshow(np.transpose(rgb, (1, 0, 2)), origin="lower")
        axes[row, 2].set_title(f"{name} — GT overlay")
        for c in range(3):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])
    fig.suptitle("Worst case (top) vs median case (bottom)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    worst_id = "BRATS_077"
    median_id = "BRATS_425"

    worst_img, worst_lbl = load_case(
        DATA_DIR / "imagesTr" / f"{worst_id}.nii.gz",
        DATA_DIR / "labelsTr" / f"{worst_id}.nii.gz",
    )
    median_img, median_lbl = load_case(
        DATA_DIR / "imagesTr" / f"{median_id}.nii.gz",
        DATA_DIR / "labelsTr" / f"{median_id}.nii.gz",
    )

    log.info(f"{worst_id}: shape {worst_img.shape}, label {worst_lbl.shape}")

    worst_stats = {
        "shape": list(worst_img.shape),
        "regions": region_stats(worst_img, worst_lbl),
        "contrast_per_modality": contrast_ratio(worst_img, worst_lbl),
        "components": connected_component_profile(worst_lbl),
        "location": volume_center_offset(worst_lbl),
    }
    median_stats = {
        "shape": list(median_img.shape),
        "regions": region_stats(median_img, median_lbl),
        "contrast_per_modality": contrast_ratio(median_img, median_lbl),
        "components": connected_component_profile(median_lbl),
        "location": volume_center_offset(median_lbl),
    }

    # Cohort-level WT-size percentile
    log.info("Computing cohort-level WT-size percentiles")
    val_ds = MSDBrainTumorDataset(root_dir=DATA_DIR, split="val", val_split=0.2, seed=42)
    wt_sizes = []
    for d in val_ds.data_dicts:
        try:
            lbl = nib.load(d["label"]).get_fdata().astype(np.int16)
            wt_sizes.append(int(np.isin(lbl, REGION_SPEC["WT"]).sum()))
        except Exception as e:
            log.warning(f"skip {d['label']}: {e}")

    worst_pct = cohort_size_percentile(worst_stats["regions"]["WT"]["voxels"], wt_sizes)
    median_pct = cohort_size_percentile(median_stats["regions"]["WT"]["voxels"], wt_sizes)
    log.info(f"  {worst_id} WT size percentile: {worst_pct:.1f}")
    log.info(f"  {median_id} WT size percentile: {median_pct:.1f}")

    worst_stats["wt_size_percentile_in_val"] = worst_pct
    median_stats["wt_size_percentile_in_val"] = median_pct

    # Compose interpretation
    wt_w = worst_stats["regions"]["WT"]["voxels"]
    wt_m = median_stats["regions"]["WT"]["voxels"]
    tc_w = worst_stats["regions"]["TC"]["voxels"]
    et_w = worst_stats["regions"]["ET"]["voxels"]

    reasons = []
    if worst_pct < 20:
        reasons.append(
            f"{worst_id} has a very small whole tumor "
            f"({wt_w} voxels, {worst_pct:.1f}th percentile in val cohort, "
            f"vs {wt_m} voxels for median case at {median_pct:.1f}th percentile). "
            "Small lesions produce a steeper Dice penalty per mis-classified voxel."
        )
    if tc_w / max(wt_w, 1) < 0.2:
        reasons.append(
            f"Tumor core is disproportionately small "
            f"({tc_w}/{wt_w} = {tc_w / max(wt_w, 1):.1%} of WT). "
            f"This matches the project-wide failure pattern: bottom-5 cases drop most on TC."
        )
    worst_contrast = worst_stats["contrast_per_modality"]
    median_contrast = median_stats["contrast_per_modality"]
    for m in range(4):
        wc = worst_contrast[f"mod{m}"]
        mc = median_contrast[f"mod{m}"]
        if wc is not None and mc is not None and wc < 0.5 * mc:
            reasons.append(
                f"Modality {m} contrast is {wc:.2f} for {worst_id} vs {mc:.2f} "
                f"for the median case — weaker tumor-vs-brain signal."
            )
    comp_w = worst_stats["components"]["WT"]
    comp_m = median_stats["components"]["WT"]
    if comp_w["num_components"] > 2 * max(comp_m["num_components"], 1):
        reasons.append(
            f"More fragmented tumor ({comp_w['num_components']} WT components "
            f"vs {comp_m['num_components']} for median)."
        )
    if et_w == 0:
        reasons.append(
            "Enhancing-tumor volume is zero in GT — Dice on ET is undefined / penalised."
        )

    diagnosis = {
        "worst_subject": worst_id,
        "worst_stats": worst_stats,
        "median_subject": median_id,
        "median_stats": median_stats,
        "likely_causes": reasons or ["No single dominant cause — likely a combination of subtle factors."],
    }
    out_path = RESULTS_DIR / "worst_case_diagnosis.json"
    out_path.write_text(json.dumps(diagnosis, indent=2))
    log.info(f"Wrote {out_path}")
    for r in reasons:
        log.info(f"  - {r}")

    # Comparison figure
    render_comparison(
        (worst_img, worst_lbl, worst_id),
        (median_img, median_lbl, median_id),
        FIG_DIR / "worst_case_comparison.png",
    )
    log.info(f"Wrote figure {FIG_DIR / 'worst_case_comparison.png'}")


if __name__ == "__main__":
    main()
