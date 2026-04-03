"""Automated RECIST 1.1 measurement from segmentation masks."""

import numpy as np
from scipy import ndimage


class RECISTMeasurer:
    """Compute RECIST 1.1 measurements from 3D binary segmentation masks.

    RECIST 1.1 criteria:
        - Complete Response (CR): Disappearance of all target lesions
        - Partial Response (PR): >= 30% decrease in sum of longest diameters
        - Progressive Disease (PD): >= 20% increase in sum of longest diameters
        - Stable Disease (SD): Neither PR nor PD criteria met
    """

    CR_THRESHOLD = 0.0  # Complete disappearance
    PR_THRESHOLD = -0.30  # 30% decrease
    PD_THRESHOLD = 0.20  # 20% increase

    def longest_axial_diameter(
        self, mask: np.ndarray, pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> float:
        """Compute longest axial diameter of a lesion from its 3D mask.

        Finds the axial slice with the largest cross-section, then computes
        the maximum Feret diameter on that slice.

        Args:
            mask: Binary 3D mask [H, W, D]
            pixdim: Voxel spacing in mm (H, W, D)

        Returns:
            Longest axial diameter in mm.
        """
        if mask.sum() == 0:
            return 0.0

        # Find axial slice with largest tumor area
        slice_areas = mask.sum(axis=(0, 1))  # Sum over H, W for each D slice
        best_slice = int(np.argmax(slice_areas))

        axial_mask = mask[:, :, best_slice]

        if axial_mask.sum() == 0:
            return 0.0

        # Get boundary coordinates
        coords = np.argwhere(axial_mask > 0)

        if len(coords) < 2:
            return 0.0

        # Compute pairwise distances (scaled by pixel spacing)
        max_dist = 0.0
        scaled_coords = coords.astype(float) * np.array([pixdim[0], pixdim[1]])

        for i in range(len(scaled_coords)):
            dists = np.sqrt(np.sum((scaled_coords[i:] - scaled_coords[i]) ** 2, axis=1))
            max_dist = max(max_dist, dists.max())

        return float(max_dist)

    def volume_mm3(
        self, mask: np.ndarray, pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> float:
        """Compute lesion volume in mm^3."""
        voxel_vol = pixdim[0] * pixdim[1] * pixdim[2]
        return float(mask.sum() * voxel_vol)

    def measure_lesions(
        self, mask: np.ndarray, pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> list[dict]:
        """Detect and measure individual lesions in a multi-lesion mask.

        Args:
            mask: Binary 3D mask (may contain multiple connected components)
            pixdim: Voxel spacing in mm

        Returns:
            List of dicts with lesion measurements, sorted by size (largest first).
        """
        labeled_array, num_features = ndimage.label(mask > 0)
        lesions = []

        for i in range(1, num_features + 1):
            lesion_mask = (labeled_array == i).astype(np.uint8)
            lesions.append(
                {
                    "id": i,
                    "longest_diameter_mm": self.longest_axial_diameter(lesion_mask, pixdim),
                    "volume_mm3": self.volume_mm3(lesion_mask, pixdim),
                    "voxel_count": int(lesion_mask.sum()),
                }
            )

        lesions.sort(key=lambda x: x["volume_mm3"], reverse=True)
        return lesions
