"""Treatment response classification based on RECIST 1.1 criteria."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.response.recist import RECISTMeasurer


class ResponseCategory(Enum):
    """RECIST 1.1 response categories."""
    CR = "Complete Response"
    PR = "Partial Response"
    SD = "Stable Disease"
    PD = "Progressive Disease"


@dataclass
class ResponseResult:
    """Result of treatment response assessment."""
    category: ResponseCategory
    baseline_sum_ld: float       # Sum of longest diameters at baseline (mm)
    followup_sum_ld: float       # Sum of longest diameters at follow-up (mm)
    percent_change: float         # Percentage change
    baseline_volume: float        # Total tumor volume at baseline (mm^3)
    followup_volume: float        # Total tumor volume at follow-up (mm^3)
    volume_change: float          # Percentage volume change
    num_baseline_lesions: int
    num_followup_lesions: int
    new_lesions: bool


class ResponseClassifier:
    """Classify treatment response from baseline and follow-up segmentation masks."""

    def __init__(self):
        self.measurer = RECISTMeasurer()

    def classify(
        self,
        baseline_mask: np.ndarray,
        followup_mask: np.ndarray,
        pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> ResponseResult:
        """Classify treatment response per RECIST 1.1.

        Args:
            baseline_mask: Binary 3D mask from baseline scan
            followup_mask: Binary 3D mask from follow-up scan
            pixdim: Voxel spacing in mm

        Returns:
            ResponseResult with category and measurements.
        """
        baseline_lesions = self.measurer.measure_lesions(baseline_mask, pixdim)
        followup_lesions = self.measurer.measure_lesions(followup_mask, pixdim)

        baseline_sum_ld = sum(les["longest_diameter_mm"] for les in baseline_lesions)
        followup_sum_ld = sum(les["longest_diameter_mm"] for les in followup_lesions)

        baseline_vol = sum(les["volume_mm3"] for les in baseline_lesions)
        followup_vol = sum(les["volume_mm3"] for les in followup_lesions)

        # Percent change in sum of longest diameters
        if baseline_sum_ld > 0:
            pct_change = (followup_sum_ld - baseline_sum_ld) / baseline_sum_ld
        else:
            pct_change = 0.0 if followup_sum_ld == 0 else float("inf")

        vol_change = (
            (followup_vol - baseline_vol) / baseline_vol if baseline_vol > 0 else 0.0
        )

        # Check for new lesions (more lesions in follow-up)
        new_lesions = len(followup_lesions) > len(baseline_lesions)

        # RECIST 1.1 classification
        if followup_sum_ld == 0 and len(followup_lesions) == 0:
            category = ResponseCategory.CR
        elif pct_change <= RECISTMeasurer.PR_THRESHOLD:
            category = ResponseCategory.PR
        elif pct_change >= RECISTMeasurer.PD_THRESHOLD or new_lesions:
            category = ResponseCategory.PD
        else:
            category = ResponseCategory.SD

        return ResponseResult(
            category=category,
            baseline_sum_ld=baseline_sum_ld,
            followup_sum_ld=followup_sum_ld,
            percent_change=pct_change,
            baseline_volume=baseline_vol,
            followup_volume=followup_vol,
            volume_change=vol_change,
            num_baseline_lesions=len(baseline_lesions),
            num_followup_lesions=len(followup_lesions),
            new_lesions=new_lesions,
        )
