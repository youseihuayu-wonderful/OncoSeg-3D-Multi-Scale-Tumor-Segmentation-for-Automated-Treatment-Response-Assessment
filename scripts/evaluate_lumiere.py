"""Longitudinal RECIST validation on the LUMIERE dataset.

For every LUMIERE patient with a baseline + at least one follow-up timepoint,
this script:

    1. Loads the four co-registered modalities (T1, CT1, T2, FLAIR).
    2. Runs the trained OncoSeg model via sliding-window inference.
    3. Extracts the enhancing-tumor (ET) channel, the region RECIST 1.1
       targets in neuro-oncology.
    4. Measures sum of longest diameters at each timepoint and classifies
       the response relative to baseline as CR / PR / SD / PD.
    5. Compares the predicted response to the expert RANO rating from
       LumiereClinicalData.csv.

Outputs under ``experiments/lumiere_results/``:
    - ``per_visit.csv``      — one row per follow-up visit
    - ``summary.json``       — agreement metrics (accuracy, Cohen kappa)
    - ``confusion_matrix.png`` — RECIST (pred) × RANO (expert)
    - ``run.log``            — per-patient log

Run from project root::

    python scripts/evaluate_lumiere.py \
        --lumiere-root /path/to/LUMIERE \
        --checkpoint experiments/local_results/oncoseg_best.pth

The checkpoint was produced by ``train_all.py`` whose inline ``OncoSeg`` class
differs from ``src/models/oncoseg.py``; we import from ``train_all`` to stay
compatible with the saved weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.lumiere import (  # noqa: E402
    ONCOSEG_MODALITIES,
    LumierePatient,
    LumiereTimepoint,
    discover_lumiere,
    patients_with_followup,
)
from src.inference import Predictor  # noqa: E402
from src.response.classifier import ResponseCategory, ResponseClassifier  # noqa: E402

log = logging.getLogger("evaluate_lumiere")


# RANO → RECIST category map. LUMIERE clinical CSV uses RANO terminology
# that happens to overlap 1:1 with RECIST categories for the four-class
# system we're validating. Lower-case and strip before lookup.
RANO_TO_RECIST: dict[str, ResponseCategory] = {
    "cr": ResponseCategory.CR,
    "complete response": ResponseCategory.CR,
    "pr": ResponseCategory.PR,
    "partial response": ResponseCategory.PR,
    "sd": ResponseCategory.SD,
    "stable disease": ResponseCategory.SD,
    "pd": ResponseCategory.PD,
    "progressive disease": ResponseCategory.PD,
    "progression": ResponseCategory.PD,
}

# Timepoints with these RANO labels are not usable as ground-truth response
# ratings (baseline has no prior scan to compare against; pre-RT scans are
# pre-treatment). They are still processed so the baseline sum-LD is
# computed, but excluded from the agreement metrics.
NON_RESPONSE_LABELS: set[str] = {"baseline", "pre-rt", "pre_rt", "prert", ""}


@dataclass
class VisitRecord:
    """One follow-up visit, with predicted and expert response."""

    patient_id: str
    baseline_tp: str
    followup_tp: str
    baseline_sum_ld_mm: float
    followup_sum_ld_mm: float
    percent_change: float
    num_baseline_lesions: int
    num_followup_lesions: int
    predicted: str
    expert_rano: str | None
    expert_rationale: str | None
    agree: bool | None  # True/False if comparable, None if expert label missing


def build_inference_transforms(
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Inference-only transforms for LUMIERE (no label key).

    Mirrors ``src/data/transforms.get_val_transforms`` but drops the label
    key since LUMIERE is used for longitudinal response validation, not
    voxel-wise GT evaluation.
    """
    keys = list(ONCOSEG_MODALITIES)
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear",) * len(keys)),
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
            CropForegroundd(keys=keys, source_key="t1c"),
            EnsureTyped(keys=keys),
        ]
    )


def load_oncoseg(checkpoint_path: Path, device: torch.device):
    """Instantiate train_all.OncoSeg and load the checkpoint.

    The import is local because ``train_all`` pulls in MONAI networks on
    import, which is slow and unnecessary when only the loader is imported
    (e.g., by the unit tests).
    """
    from train_all import OncoSeg  # noqa: PLC0415 — see docstring

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
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def segment_timepoint(
    predictor: Predictor,
    transforms: Compose,
    tp: LumiereTimepoint,
) -> np.ndarray:
    """Run OncoSeg on one LUMIERE timepoint and return the ET binary mask."""
    data = tp.as_data_dict()
    # Predictor.predict_volume expects an already-preprocessed [1, C, H, W, D] tensor.
    processed = transforms(data)
    stacked = torch.stack([processed[m] for m in ONCOSEG_MODALITIES], dim=0)
    # EnsureChannelFirstd gives [1, H, W, D] per modality; squeeze the channel
    # dim before re-stacking so we end up with [C, H, W, D], then add batch dim.
    if stacked.ndim == 5 and stacked.shape[1] == 1:
        stacked = stacked.squeeze(1)
    image = stacked.unsqueeze(0)
    result = predictor.predict_volume(image)
    seg = result["segmentation"]  # [3, H, W, D] channel-first sigmoid output
    if seg.ndim != 4 or seg.shape[0] < 3:
        raise RuntimeError(
            f"Unexpected segmentation shape {seg.shape} for {tp.patient_id}/{tp.timepoint_id}; "
            "expected [3, H, W, D] from sigmoid output."
        )
    return seg[2].astype(np.uint8)


def normalise_rano(label: str | None) -> str | None:
    """Lower-case + strip a RANO label, treating empty/none as missing."""
    if label is None:
        return None
    trimmed = label.strip().lower()
    return trimmed or None


def rano_to_recist(label: str | None) -> ResponseCategory | None:
    """Map a RANO string to a RECIST category, or None if not a response rating."""
    norm = normalise_rano(label)
    if norm is None or norm in NON_RESPONSE_LABELS:
        return None
    return RANO_TO_RECIST.get(norm)


def evaluate_patient(
    patient: LumierePatient,
    predictor: Predictor,
    transforms: Compose,
    classifier: ResponseClassifier,
    pixdim: tuple[float, float, float],
) -> list[VisitRecord]:
    """Run OncoSeg on every timepoint and classify each follow-up."""
    baseline = patient.baseline()
    if baseline is None:
        return []

    log.info("Patient %s: baseline=%s, %d follow-ups",
             patient.patient_id, baseline.timepoint_id, len(patient.followups()))

    baseline_mask = segment_timepoint(predictor, transforms, baseline)

    records: list[VisitRecord] = []
    for fu in patient.followups():
        try:
            followup_mask = segment_timepoint(predictor, transforms, fu)
        except Exception as exc:  # pragma: no cover — logged and skipped
            log.error("Inference failed on %s/%s: %s", patient.patient_id, fu.timepoint_id, exc)
            continue

        if baseline_mask.shape != followup_mask.shape:
            # LUMIERE scans within a patient are co-registered; a shape mismatch
            # indicates the transforms pipeline cropped the foreground differently.
            # Pad the smaller to the larger on each axis with zeros so RECIST
            # can still compare sum-of-longest-diameters fairly.
            baseline_mask, followup_mask = _pad_to_common(baseline_mask, followup_mask)

        result = classifier.classify(baseline_mask, followup_mask, pixdim=pixdim)

        expert = rano_to_recist(fu.rano)
        agree: bool | None
        if expert is None:
            agree = None
        else:
            agree = result.category == expert

        records.append(
            VisitRecord(
                patient_id=patient.patient_id,
                baseline_tp=baseline.timepoint_id,
                followup_tp=fu.timepoint_id,
                baseline_sum_ld_mm=result.baseline_sum_ld,
                followup_sum_ld_mm=result.followup_sum_ld,
                percent_change=result.percent_change,
                num_baseline_lesions=result.num_baseline_lesions,
                num_followup_lesions=result.num_followup_lesions,
                predicted=result.category.name,
                expert_rano=fu.rano,
                expert_rationale=fu.rano_rationale,
                agree=agree,
            )
        )
    return records


def _pad_to_common(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad two 3D masks to their element-wise max shape."""
    target = tuple(max(sa, sb) for sa, sb in zip(a.shape, b.shape, strict=True))

    def pad(mask: np.ndarray) -> np.ndarray:
        pad_width = [(0, t - s) for s, t in zip(mask.shape, target, strict=True)]
        return np.pad(mask, pad_width, mode="constant", constant_values=0)

    return pad(a), pad(b)


def cohen_kappa(y_true: list[str], y_pred: list[str]) -> float:
    """Compute Cohen's kappa between two parallel string-label sequences."""
    if not y_true:
        return float("nan")
    labels = sorted(set(y_true) | set(y_pred))
    idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    conf = np.zeros((n, n), dtype=np.float64)
    for t, p in zip(y_true, y_pred, strict=True):
        conf[idx[t], idx[p]] += 1
    total = conf.sum()
    if total == 0:
        return float("nan")
    observed = np.trace(conf) / total
    expected = ((conf.sum(axis=0) / total) * (conf.sum(axis=1) / total)).sum()
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return float((observed - expected) / (1 - expected))


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> np.ndarray:
    """Build a confusion matrix with rows=true, cols=pred."""
    idx = {label: i for i, label in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        if t in idx and p in idx:
            mat[idx[t], idx[p]] += 1
    return mat


def plot_confusion_matrix(
    mat: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str = "LUMIERE RECIST agreement (expert RANO × OncoSeg prediction)",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("OncoSeg predicted")
    ax.set_ylabel("Expert RANO")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            colour = "white" if mat[i, j] > mat.max() / 2 else "black"
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", color=colour)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarise(records: list[VisitRecord]) -> dict:
    """Build the summary.json payload from per-visit records."""
    comparable = [r for r in records if r.agree is not None]
    total_visits = len(records)
    total_comparable = len(comparable)

    if total_comparable == 0:
        return {
            "total_visits": total_visits,
            "comparable_visits": 0,
            "accuracy": None,
            "cohen_kappa": None,
            "confusion_matrix": None,
            "labels": [c.name for c in ResponseCategory],
            "predicted_distribution": dict(Counter(r.predicted for r in records)),
            "note": "No follow-up visits carried an expert RANO rating — agreement metrics unavailable.",
        }

    labels = [c.name for c in ResponseCategory]
    y_true = [(rano_to_recist(r.expert_rano).name if rano_to_recist(r.expert_rano) else "UNKNOWN") for r in comparable]
    y_pred = [r.predicted for r in comparable]
    mat = confusion_matrix(y_true, y_pred, labels)
    correct = sum(1 for r in comparable if r.agree)
    accuracy = correct / total_comparable
    kappa = cohen_kappa(y_true, y_pred)

    return {
        "total_visits": total_visits,
        "comparable_visits": total_comparable,
        "accuracy": accuracy,
        "cohen_kappa": kappa,
        "confusion_matrix": mat.tolist(),
        "labels": labels,
        "predicted_distribution": dict(Counter(r.predicted for r in records)),
        "expert_distribution": dict(Counter(y_true)),
    }


def write_per_visit_csv(records: list[VisitRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("")
        return
    fieldnames = list(asdict(records[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))


def run(
    lumiere_root: Path,
    checkpoint: Path,
    output_dir: Path,
    max_patients: int | None = None,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
    mc_samples: int = 0,
) -> dict:
    """End-to-end LUMIERE evaluation. Returns the summary dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(file_handler)

    log.info("LUMIERE root: %s", lumiere_root)
    log.info("Checkpoint:   %s", checkpoint)
    log.info("Output:       %s", output_dir)

    patients = patients_with_followup(discover_lumiere(lumiere_root))
    if max_patients is not None:
        patients = patients[:max_patients]
    log.info("Evaluating %d patients (with at least one follow-up)", len(patients))

    device = get_device()
    log.info("Device: %s", device)
    model = load_oncoseg(checkpoint, device)
    predictor = Predictor(
        model=model,
        device=device,
        roi_size=roi_size,
        mc_samples=mc_samples,
    )
    transforms = build_inference_transforms(pixdim=pixdim)
    classifier = ResponseClassifier()

    all_records: list[VisitRecord] = []
    for patient in patients:
        try:
            all_records.extend(
                evaluate_patient(patient, predictor, transforms, classifier, pixdim)
            )
        except Exception as exc:  # pragma: no cover — logged and skipped
            log.error("Patient %s failed: %s", patient.patient_id, exc)
            continue

    write_per_visit_csv(all_records, output_dir / "per_visit.csv")
    summary = summarise(all_records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if summary.get("confusion_matrix") is not None:
        plot_confusion_matrix(
            np.array(summary["confusion_matrix"], dtype=np.int64),
            summary["labels"],
            output_dir / "confusion_matrix.png",
        )

    log.info(
        "Done. Visits=%d  Comparable=%d  Accuracy=%s  Kappa=%s",
        summary["total_visits"],
        summary["comparable_visits"],
        summary["accuracy"],
        summary["cohen_kappa"],
    )
    log.removeHandler(file_handler)
    file_handler.close()
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lumiere-root", type=Path, required=True,
                        help="Path to the LUMIERE dataset root (contains Patient-XX/ subdirs).")
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_ROOT / "experiments" / "local_results" / "oncoseg_best.pth",
                        help="Path to trained OncoSeg checkpoint (from train_all.py).")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "experiments" / "lumiere_results",
                        help="Where to write per_visit.csv, summary.json, confusion_matrix.png.")
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Limit to first N patients (for smoke tests).")
    parser.add_argument("--roi", type=int, default=96,
                        help="Sliding-window ROI size (cubic). Default 96.")
    parser.add_argument("--mc-samples", type=int, default=0,
                        help="MC Dropout samples per timepoint (0 = deterministic).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    roi = (args.roi, args.roi, args.roi)
    summary = run(
        lumiere_root=args.lumiere_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        max_patients=args.max_patients,
        roi_size=roi,
        mc_samples=args.mc_samples,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
