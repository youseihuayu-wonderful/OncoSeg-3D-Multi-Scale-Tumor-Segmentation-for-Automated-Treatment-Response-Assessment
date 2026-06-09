"""Write OncoSeg segmentation masks as standard **DICOM SEG** objects.

DICOM SEG is the interoperable way to ship an AI segmentation back to a PACS so a
radiologist can review it overlaid on the original study in any standard viewer
(OHIF, 3D Slicer, ...). This is L3 step 1 — see ``docs/L3_Integration_Plan.md``.

The OncoSeg model emits three BraTS regions that *overlap* (ET ⊂ TC ⊂ WT), so each
is written as its own **binary segment** rather than collapsed into a single label
map (which cannot represent overlap).
"""

from __future__ import annotations

from collections.abc import Sequence

import highdicom as hd
import numpy as np
from pydicom import Dataset
from pydicom.sr.codedict import codes

# OncoSeg's three output channels, in channel order (TC, WT, ET).
ONCOSEG_SEGMENT_LABELS: tuple[str, ...] = ("Tumor Core", "Whole Tumor", "Enhancing Tumor")


def _segment_descriptions(
    labels: Sequence[str], algorithm_version: str
) -> list[hd.seg.SegmentDescription]:
    """One SegmentDescription per channel, all typed as automatically-segmented neoplasm."""
    algorithm = hd.AlgorithmIdentificationSequence(
        name="OncoSeg",
        family=codes.DCM.ArtificialIntelligence,
        version=algorithm_version,
    )
    return [
        hd.seg.SegmentDescription(
            segment_number=i,
            segment_label=label,
            # SNOMED: a morphologically abnormal structure that is a neoplasm (tumor).
            segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
            segmented_property_type=codes.SCT.Neoplasm,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algorithm,
        )
        for i, label in enumerate(labels, start=1)
    ]


def mask_to_dicom_seg(
    source_images: Sequence[Dataset],
    mask: np.ndarray,
    *,
    segment_labels: Sequence[str] = ONCOSEG_SEGMENT_LABELS,
    series_number: int = 100,
    instance_number: int = 1,
    series_description: str = "OncoSeg AI Segmentation",
    algorithm_version: str = "1.0.0",
) -> hd.seg.Segmentation:
    """Build a DICOM SEG from a segmentation mask and its source DICOM slices.

    Args:
        source_images: the original DICOM slices the mask was computed on, as
            pydicom ``Dataset``s, ordered to match ``mask``'s frame axis.
        mask: binary segmentation. Either ``(frames, rows, cols)`` for a single
            segment, or ``(frames, rows, cols, n_segments)`` one-hot for multiple
            overlapping segments. Non-zero = foreground.
        segment_labels: human-readable label per segment; length must equal the
            number of segments in ``mask``.
        series_number / instance_number / series_description: identify the new
            derived SEG series.
        algorithm_version: recorded as the producing algorithm's version.

    Returns:
        A ``highdicom.seg.Segmentation`` dataset, ready to ``.save_as(path)`` or
        push to a PACS via STOW-RS.

    Raises:
        ValueError: on shape/segment-count/frame-count mismatch.
    """
    source_images = list(source_images)
    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = mask[..., np.newaxis]
    if mask.ndim != 4:
        raise ValueError(
            f"mask must be 3D (frames, rows, cols) or 4D (..., segments); got {mask.ndim}D"
        )

    n_segments = mask.shape[-1]
    if n_segments != len(segment_labels):
        raise ValueError(
            f"mask has {n_segments} segment(s) but {len(segment_labels)} label(s) were given"
        )
    if mask.shape[0] != len(source_images):
        raise ValueError(
            f"mask has {mask.shape[0]} frame(s) but {len(source_images)} source image(s) "
            "were given; they must correspond one-to-one"
        )

    binary_mask = (mask != 0).astype(np.uint8)

    return hd.seg.Segmentation(
        source_images=source_images,
        pixel_array=binary_mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=_segment_descriptions(segment_labels, algorithm_version),
        series_instance_uid=hd.UID(),
        series_number=series_number,
        sop_instance_uid=hd.UID(),
        instance_number=instance_number,
        manufacturer="OncoSeg",
        manufacturer_model_name="OncoSeg",
        software_versions=algorithm_version,
        device_serial_number="OncoSeg-0001",
        series_description=series_description,
    )
