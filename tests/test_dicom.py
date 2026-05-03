"""Tests for the DICOM series loader (src/data/dicom.py).

Uses pydicom to synthesize a minimal series in a tmp dir, then exercises
load_dicom_series / load_dicom_zip end-to-end.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from src.data.dicom import DICOMLoadError, load_dicom_series, load_dicom_zip


def _make_slice(
    rows: int,
    cols: int,
    z_mm: float,
    series_uid: str,
    pixel_array: np.ndarray,
    *,
    pixel_spacing=(1.0, 1.0),
    slice_thickness: float = 1.0,
    orientation_lps=(1, 0, 0, 0, 1, 0),
    rescale_slope: float | None = None,
    rescale_intercept: float | None = None,
) -> Dataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta

    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "CT"
    ds.PatientID = "TEST"
    ds.PatientName = "Test^Patient"

    ds.Rows = rows
    ds.Columns = cols
    ds.PixelSpacing = list(pixel_spacing)
    ds.SliceThickness = slice_thickness
    ds.ImageOrientationPatient = list(orientation_lps)
    ds.ImagePositionPatient = [0.0, 0.0, z_mm]
    ds.SliceLocation = z_mm

    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 1  # signed

    if rescale_slope is not None:
        ds.RescaleSlope = rescale_slope
    if rescale_intercept is not None:
        ds.RescaleIntercept = rescale_intercept

    ds.PixelData = pixel_array.astype(np.int16).tobytes()
    return ds


def _write_series(
    dest: Path,
    volume: np.ndarray,
    *,
    pixel_spacing=(1.0, 1.0),
    slice_thickness: float = 1.0,
    rescale_slope: float | None = None,
    rescale_intercept: float | None = None,
) -> Path:
    """Write `volume` of shape (rows, cols, n_slices) as a DICOM series under `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    rows, cols, n_slices = volume.shape
    for i in range(n_slices):
        ds = _make_slice(
            rows=rows,
            cols=cols,
            z_mm=i * slice_thickness,
            series_uid=series_uid,
            pixel_array=volume[:, :, i],
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            rescale_slope=rescale_slope,
            rescale_intercept=rescale_intercept,
        )
        ds.InstanceNumber = i + 1
        ds.save_as(dest / f"slice_{i:03d}.dcm", write_like_original=False)
    return dest


class TestLoadDicomSeries:
    def test_round_trips_volume(self, tmp_path: Path):
        volume = np.arange(4 * 5 * 3, dtype=np.int16).reshape(4, 5, 3)
        _write_series(tmp_path / "series", volume)

        loaded, affine, pixdim = load_dicom_series(tmp_path / "series")
        assert loaded.shape == volume.shape
        np.testing.assert_array_equal(loaded.astype(np.int16), volume)
        assert pixdim == (1.0, 1.0, 1.0)
        assert affine.shape == (4, 4)

    def test_applies_rescale_slope_intercept(self, tmp_path: Path):
        raw = np.full((2, 2, 2), 100, dtype=np.int16)
        _write_series(
            tmp_path / "series", raw,
            rescale_slope=2.0, rescale_intercept=-50.0,
        )
        loaded, _, _ = load_dicom_series(tmp_path / "series")
        # 100 * 2 - 50 = 150 everywhere
        assert np.allclose(loaded, 150.0)

    def test_anisotropic_pixdim(self, tmp_path: Path):
        vol = np.zeros((3, 3, 4), dtype=np.int16)
        _write_series(
            tmp_path / "series", vol,
            pixel_spacing=(0.8, 1.2),  # (row, col) mm
            slice_thickness=3.0,
        )
        _, _, pixdim = load_dicom_series(tmp_path / "series")
        # Loader returns (col_spacing, row_spacing, slice_gap)
        assert pixdim[0] == pytest.approx(1.2)
        assert pixdim[1] == pytest.approx(0.8)
        assert pixdim[2] == pytest.approx(3.0)

    def test_slices_sorted_by_position(self, tmp_path: Path):
        """Scrambled filenames still yield a correctly-ordered volume."""
        dest = tmp_path / "series"
        dest.mkdir()
        volume = np.stack(
            [np.full((2, 2), v, dtype=np.int16) for v in [10, 20, 30]], axis=-1
        )
        series_uid = generate_uid()
        # Write in reverse z order with lexically earliest name for highest z
        names_by_z = {0: "slice_zzz.dcm", 1: "slice_mmm.dcm", 2: "slice_aaa.dcm"}
        for z in [0, 1, 2]:
            ds = _make_slice(2, 2, float(z), series_uid, volume[:, :, z])
            ds.InstanceNumber = z + 1
            ds.save_as(dest / names_by_z[z], write_like_original=False)

        loaded, _, _ = load_dicom_series(dest)
        np.testing.assert_array_equal(loaded[:, :, 0], np.full((2, 2), 10))
        np.testing.assert_array_equal(loaded[:, :, 1], np.full((2, 2), 20))
        np.testing.assert_array_equal(loaded[:, :, 2], np.full((2, 2), 30))

    def test_mixed_series_rejected(self, tmp_path: Path):
        dest = tmp_path / "series"
        dest.mkdir()
        uid_a, uid_b = generate_uid(), generate_uid()
        vol = np.zeros((2, 2), dtype=np.int16)
        _make_slice(2, 2, 0.0, uid_a, vol).save_as(dest / "a.dcm", write_like_original=False)
        _make_slice(2, 2, 1.0, uid_b, vol).save_as(dest / "b.dcm", write_like_original=False)

        with pytest.raises(DICOMLoadError, match="multiple SeriesInstanceUID"):
            load_dicom_series(dest)

    def test_empty_dir_rejected(self, tmp_path: Path):
        with pytest.raises(DICOMLoadError, match="No DICOM files"):
            load_dicom_series(tmp_path)

    def test_missing_source_rejected(self, tmp_path: Path):
        with pytest.raises(DICOMLoadError, match="does not exist"):
            load_dicom_series(tmp_path / "nope")

    def test_inconsistent_shape_rejected(self, tmp_path: Path):
        dest = tmp_path / "series"
        dest.mkdir()
        uid = generate_uid()
        _make_slice(4, 4, 0.0, uid, np.zeros((4, 4), dtype=np.int16)).save_as(
            dest / "a.dcm", write_like_original=False
        )
        _make_slice(5, 5, 1.0, uid, np.zeros((5, 5), dtype=np.int16)).save_as(
            dest / "b.dcm", write_like_original=False
        )
        with pytest.raises(DICOMLoadError, match="Inconsistent slice shape"):
            load_dicom_series(dest)


class TestLoadDicomZip:
    def test_extracts_and_loads(self, tmp_path: Path):
        series_dir = tmp_path / "src"
        vol = np.full((3, 3, 4), 42, dtype=np.int16)
        _write_series(series_dir, vol)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for path in series_dir.iterdir():
                zf.write(path, arcname=path.name)

        workdir = tmp_path / "work"
        workdir.mkdir()
        loaded, _, _ = load_dicom_zip(buf.getvalue(), workdir)
        assert loaded.shape == vol.shape
        np.testing.assert_array_equal(loaded.astype(np.int16), vol)
