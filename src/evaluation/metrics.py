"""Segmentation evaluation metrics for tumor segmentation."""

import torch
from monai.metrics import (
    ConfusionMatrixMetric,
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
)


class SegmentationMetrics:
    """Comprehensive segmentation metrics for BraTS evaluation.

    Computes per-region metrics for:
        - Enhancing Tumor (ET)
        - Tumor Core (TC)
        - Whole Tumor (WT)

    Metrics:
        - Dice Score
        - Hausdorff Distance 95%
        - Average Surface Distance
        - Sensitivity (Recall)
        - Specificity
    """

    REGION_NAMES = ["ET", "TC", "WT"]

    def __init__(self):
        self.dice = DiceMetric(include_background=True, reduction="mean_batch")
        self.hd95 = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean_batch"
        )
        self.asd = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean_batch"
        )
        self.confusion = ConfusionMatrixMetric(
            include_background=True,
            metric_name=["sensitivity", "specificity"],
            reduction="mean_batch",
        )

    def reset(self):
        """Reset all metric accumulators."""
        self.dice.reset()
        self.hd95.reset()
        self.asd.reset()
        self.confusion.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with a batch of predictions.

        Args:
            pred: Binary predictions [B, C, H, W, D]
            target: Ground truth [B, C, H, W, D]
        """
        self.dice(y_pred=pred, y=target)
        self.hd95(y_pred=pred, y=target)
        self.asd(y_pred=pred, y=target)
        self.confusion(y_pred=pred, y=target)

    def compute(self) -> dict[str, float]:
        """Compute all metrics and return as a flat dictionary."""
        dice_scores = self.dice.aggregate()
        hd95_scores = self.hd95.aggregate()
        asd_scores = self.asd.aggregate()
        confusion_scores = self.confusion.aggregate()

        results = {}

        for i, region in enumerate(self.REGION_NAMES):
            results[f"dice_{region}"] = dice_scores[i].item()
            results[f"hd95_{region}"] = hd95_scores[i].item()
            results[f"asd_{region}"] = asd_scores[i].item()
            if isinstance(confusion_scores, (list, tuple)):
                results[f"sensitivity_{region}"] = confusion_scores[0][i].item()
                results[f"specificity_{region}"] = confusion_scores[1][i].item()

        # Mean across regions
        results["dice_mean"] = dice_scores.mean().item()
        results["hd95_mean"] = hd95_scores.mean().item()

        return results

    def summary(self) -> str:
        """Return a formatted summary string."""
        results = self.compute()
        lines = ["=" * 60, "Segmentation Metrics", "=" * 60]

        for region in self.REGION_NAMES:
            lines.append(f"\n  {region}:")
            lines.append(f"    Dice:        {results[f'dice_{region}']:.4f}")
            lines.append(f"    HD95 (mm):   {results[f'hd95_{region}']:.2f}")
            lines.append(f"    ASD (mm):    {results[f'asd_{region}']:.2f}")

        lines.append(f"\n  Mean Dice: {results['dice_mean']:.4f}")
        lines.append("=" * 60)

        return "\n".join(lines)
