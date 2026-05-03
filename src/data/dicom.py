"""DICOM series loader for raw clinical CT/MRI input.

Real clinical data arrives as DICOM series (one file per slice). This module
converts a series directory into the dense 3D numpy volume + NIfTI-style affine
expected by the OncoSeg inference pipeline — without depending on dcm2niix or
other external binaries.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pydicom


class DICOMLoadError(ValueError):
    """Raised when a DICOM series cannot be assembled into a usable volume."""


def _is_dicom(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() == ".dcm":
        return True
    try:
        with path.open("rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False


def _read_slice(path: Path) -> pydicom.Dataset:
    return pydicom.dcmread(str(path))


def _slice_sort_key(ds: pydicom.Dataset) -> float:
    """Sort slices along the through-plane axis.

    ImagePositionPatient[2] (slice position along +Z in patient coords) is
    available on almost all cross-sectional DICOMs and matches what dcm2niix
    and ITK use. SliceLocation is a weaker fallback.
    """
    if "ImagePositionPatient" in ds:
        return float(ds.ImagePositionPatient[2])
    if "SliceLocation" in ds:
        return float(ds.SliceLocation)
    if "InstanceNumber" in ds:
        return float(ds.InstanceNumber)
    raise DICOMLoadError("Slice has no positional metadata (IPP/SliceLocation/InstanceNumber)")


def _build_affine(
    first: pydicom.Dataset,
    last: pydicom.Dataset,
    num_slices: int,
) -> np.ndarray:
    """Build a 4x4 NIfTI-style (RAS+) affine from DICOM headers.

    DICOM stores image orientation in LPS+ (Left, Posterior, Superior) patient
    coordinates; NIfTI expects RAS+. We flip X and Y to convert.
    """
    orient = np.asarray(first.ImageOrientationPatient, dtype=float)
    row_cos = orient[:3]
    col_cos = orient[3:]

    pixel_spacing = np.asarray(first.PixelSpacing, dtype=float)  # (row, col)
    ipp_first = np.asarray(first.ImagePositionPatient, dtype=float)

    if num_slices > 1:
        ipp_last = np.asarray(last.ImagePositionPatient, dtype=float)
        slice_vec = (ipp_last - ipp_first) / (num_slices - 1)
    else:
        thickness = float(getattr(first, "SliceThickness", 1.0))
        slice_vec = np.cross(row_cos, col_cos) * thickness

    affine = np.eye(4)
    affine[:3, 0] = row_cos * pixel_spacing[1]  # column direction (x in array)
    affine[:3, 1] = col_cos * pixel_spacing[0]  # row direction (y in array)
    affine[:3, 2] = slice_vec
    affine[:3, 3] = ipp_first

    # LPS+ -> RAS+ : negate x and y axes and the translation's x, y components.
    affine[0] *= -1
    affine[1] *= -1
    return affine


def load_dicom_series(source: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load a DICOM series from a directory into (volume, affine, pixdim).

    Volume has shape [rows, cols, slices] (matches nibabel's in-memory layout)
    with float32 dtype. RescaleSlope/Intercept are applied when present so CTs
    come back in Hounsfield Units.
    """
    if not source.exists():
        raise DICOMLoadError(f"DICOM source does not exist: {source}")

    if source.is_file():
        candidates = [source]
    else:
        candidates = sorted(p for p in source.rglob("*") if _is_dicom(p))

    if not candidates:
        raise DICOMLoadError(f"No DICOM files found under {source}")

    slices = [_read_slice(p) for p in candidates]
    slices.sort(key=_slice_sort_key)

    rows = int(slices[0].Rows)
    cols = int(slices[0].Columns)
    series_uid = getattr(slices[0], "SeriesInstanceUID", None)
    for s in slices[1:]:
        if int(s.Rows) != rows or int(s.Columns) != cols:
            raise DICOMLoadError(
                f"Inconsistent slice shape: expected {(rows, cols)} got {(s.Rows, s.Columns)}"
            )
        if series_uid and getattr(s, "SeriesInstanceUID", None) != series_uid:
            raise DICOMLoadError(
                "DICOM directory mixes multiple SeriesInstanceUID — expected one series"
            )

    volume = np.stack([np.asarray(s.pixel_array) for s in slices], axis=-1).astype(np.float32)

    slope = float(getattr(slices[0], "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        volume = volume * slope + intercept

    affine = _build_affine(slices[0], slices[-1], num_slices=len(slices))

    pix = np.asarray(slices[0].PixelSpacing, dtype=float)
    if len(slices) > 1:
        ipp0 = np.asarray(slices[0].ImagePositionPatient, dtype=float)
        ipp1 = np.asarray(slices[-1].ImagePositionPatient, dtype=float)
        slice_gap = float(np.linalg.norm(ipp1 - ipp0) / (len(slices) - 1))
    else:
        slice_gap = float(getattr(slices[0], "SliceThickness", 1.0))
    pixdim = (float(pix[1]), float(pix[0]), slice_gap)

    return volume, affine, pixdim


def load_dicom_zip(
    zip_bytes: bytes, workdir: Path
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Extract a ZIP of DICOM slices into `workdir` and load them as a volume."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(workdir)
    return load_dicom_series(workdir)
