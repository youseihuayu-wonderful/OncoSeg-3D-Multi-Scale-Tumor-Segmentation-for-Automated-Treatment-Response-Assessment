"""Failure case analysis for segmentation models."""

import numpy as np
from scipy import ndimage


class FailureAnalyzer:
    """Identify and categorize prediction failures by tumor characteristics.

    Usage:
        analyzer = FailureAnalyzer()
        for subject_id, pred, gt, dice in results:
            analyzer.add_subject(subject_id, pred, gt, dice, pixdim=(1,1,1))
        print(analyzer.failure_report())
        print(analyzer.size_stratified_analysis())
    """

    def __init__(self, dice_threshold: float = 0.5):
        self.dice_threshold = dice_threshold
        self.subjects: list[dict] = []

    def add_subject(
        self,
        subject_id: str,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        dice_scores: dict[str, float],
        pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        """Add a subject's results for analysis.

        Args:
            subject_id: Subject identifier
            pred_mask: Predicted binary mask [H, W, D]
            gt_mask: Ground truth binary mask [H, W, D]
            dice_scores: {"ET": 0.82, "TC": 0.85, "WT": 0.90}
            pixdim: Voxel spacing in mm
        """
        voxel_vol = pixdim[0] * pixdim[1] * pixdim[2]
        gt_volume = float(gt_mask.sum()) * voxel_vol
        pred_volume = float(pred_mask.sum()) * voxel_vol

        # Count connected components
        gt_components, gt_n = ndimage.label(gt_mask > 0)
        pred_components, pred_n = ndimage.label(pred_mask > 0)

        # Classify tumor size
        if gt_volume == 0:
            size_category = "no_tumor"
        elif gt_volume < 1000:
            size_category = "small"       # < 1 cm³
        elif gt_volume < 10000:
            size_category = "medium"      # 1-10 cm³
        else:
            size_category = "large"       # > 10 cm³

        # Classify failure type
        mean_dice = np.mean(list(dice_scores.values()))
        if mean_dice >= 0.8:
            failure_type = "good"
        elif mean_dice >= self.dice_threshold:
            failure_type = "moderate"
        else:
            failure_type = "failure"

        # Over/under segmentation
        if gt_volume > 0:
            volume_ratio = pred_volume / gt_volume
        else:
            volume_ratio = float("inf") if pred_volume > 0 else 1.0

        if volume_ratio > 1.5:
            seg_type = "over_segmentation"
        elif volume_ratio < 0.5:
            seg_type = "under_segmentation"
        else:
            seg_type = "balanced"

        self.subjects.append({
            "subject_id": subject_id,
            "dice_scores": dice_scores,
            "mean_dice": mean_dice,
            "gt_volume_mm3": gt_volume,
            "pred_volume_mm3": pred_volume,
            "volume_ratio": volume_ratio,
            "gt_lesion_count": gt_n,
            "pred_lesion_count": pred_n,
            "size_category": size_category,
            "failure_type": failure_type,
            "seg_type": seg_type,
        })

    def failure_report(self, top_n: int = 10) -> str:
        """Report worst-performing subjects."""
        sorted_subjects = sorted(self.subjects, key=lambda s: s["mean_dice"])

        lines = [
            f"Failure Report (worst {top_n} subjects, Dice threshold={self.dice_threshold})",
            "─" * 80,
        ]

        # Summary
        failures = [s for s in self.subjects if s["failure_type"] == "failure"]
        moderate = [s for s in self.subjects if s["failure_type"] == "moderate"]
        good = [s for s in self.subjects if s["failure_type"] == "good"]

        lines.append(f"  Total subjects: {len(self.subjects)}")
        lines.append(f"  Good (Dice≥0.8):     {len(good)} ({len(good)/max(len(self.subjects),1)*100:.0f}%)")
        lines.append(f"  Moderate (0.5-0.8):  {len(moderate)} ({len(moderate)/max(len(self.subjects),1)*100:.0f}%)")
        lines.append(f"  Failure (Dice<0.5):  {len(failures)} ({len(failures)/max(len(self.subjects),1)*100:.0f}%)")

        lines.append(f"\n  Worst {top_n} subjects:")
        for s in sorted_subjects[:top_n]:
            lines.append(
                f"    {s['subject_id']:20s} "
                f"Dice={s['mean_dice']:.3f}  "
                f"Vol={s['gt_volume_mm3']:.0f}mm³  "
                f"Size={s['size_category']:8s}  "
                f"Seg={s['seg_type']}"
            )

        return "\n".join(lines)

    def size_stratified_analysis(self) -> str:
        """Analyze performance stratified by tumor size."""
        categories = {"small": [], "medium": [], "large": [], "no_tumor": []}

        for s in self.subjects:
            categories[s["size_category"]].append(s["mean_dice"])

        lines = [
            "Size-Stratified Analysis",
            "─" * 60,
            f"  {'Category':12s} {'Count':>6s} {'Mean Dice':>10s} {'Std':>8s} {'Min':>8s} {'Max':>8s}",
        ]

        for cat in ["small", "medium", "large"]:
            scores = categories[cat]
            if not scores:
                lines.append(f"  {cat:12s} {0:>6d}        —        —        —        —")
                continue
            lines.append(
                f"  {cat:12s} {len(scores):>6d} "
                f"{np.mean(scores):>10.4f} "
                f"{np.std(scores):>8.4f} "
                f"{np.min(scores):>8.4f} "
                f"{np.max(scores):>8.4f}"
            )

        lines.append(f"\n  Small = <1cm³, Medium = 1-10cm³, Large = >10cm³")

        return "\n".join(lines)

    def segmentation_bias_analysis(self) -> str:
        """Analyze over/under segmentation patterns."""
        over = [s for s in self.subjects if s["seg_type"] == "over_segmentation"]
        under = [s for s in self.subjects if s["seg_type"] == "under_segmentation"]
        balanced = [s for s in self.subjects if s["seg_type"] == "balanced"]

        lines = [
            "Segmentation Bias Analysis",
            "─" * 60,
            f"  Over-segmentation (vol_ratio > 1.5):  {len(over):>4d} subjects",
            f"  Under-segmentation (vol_ratio < 0.5): {len(under):>4d} subjects",
            f"  Balanced (0.5-1.5):                   {len(balanced):>4d} subjects",
        ]

        if over:
            mean_ratio = np.mean([s["volume_ratio"] for s in over])
            mean_dice = np.mean([s["mean_dice"] for s in over])
            lines.append(f"\n  Over-segmented: mean ratio={mean_ratio:.2f}x, mean Dice={mean_dice:.3f}")

        if under:
            mean_ratio = np.mean([s["volume_ratio"] for s in under])
            mean_dice = np.mean([s["mean_dice"] for s in under])
            lines.append(f"  Under-segmented: mean ratio={mean_ratio:.2f}x, mean Dice={mean_dice:.3f}")

        return "\n".join(lines)
