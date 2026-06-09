"""Tests for the DICOM SEG writer — build a real SEG from a synthetic MR series.

No real patient data: we synthesize a tiny, geometrically valid MR series with
pydicom so the writer exercises the full highdicom path on every run.
"""

import sys
from pathlib import Path

import highdicom as hd
import numpy as np
import pytest
from pydicom import Dataset
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dicom.seg_writer import ONCOSEG_SEGMENT_LABELS, mask_to_dicom_seg  # noqa: E402


def _synthetic_mr_series(num_slices: int = 3, rows: int = 8, cols: int = 8) -> list[Dataset]:
    """A minimal but geometrically valid axial MR series (one series, N slices)."""
    study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    images: list[Dataset] = []
    for i in range(num_slices):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.PatientID = "ONCOSEG-TEST"
        ds.PatientName = "Test^Patient"
        # Type-2 patient/study attributes highdicom copies onto the derived SEG
        # (present-but-empty is valid DICOM; real source images always carry these).
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        ds.StudyID = "1"
        ds.StudyDate = "20240101"
        ds.StudyTime = "120000"
        ds.AccessionNumber = ""
        ds.ReferringPhysicianName = ""
        ds.Modality = "MR"
        ds.SeriesNumber = 1
        ds.InstanceNumber = i + 1
        ds.Rows = rows
        ds.Columns = cols
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.ImagePositionPatient = [0.0, 0.0, float(i)]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = np.zeros((rows, cols), dtype=np.uint16).tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        images.append(ds)
    return images


@pytest.fixture
def series():
    return _synthetic_mr_series()


def _three_channel_mask(num_slices=3, rows=8, cols=8):
    """One-hot (frames, rows, cols, 3) mask with nested TC ⊂ WT and a small ET blob."""
    mask = np.zeros((num_slices, rows, cols, 3), dtype=np.uint8)
    mask[:, 2:6, 2:6, 1] = 1  # Whole Tumor (larger)
    mask[:, 3:5, 3:5, 0] = 1  # Tumor Core (inside WT)
    mask[1, 3:4, 3:4, 2] = 1  # Enhancing Tumor (tiny, one slice)
    return mask


def test_builds_valid_three_segment_seg(series):
    seg = mask_to_dicom_seg(series, _three_channel_mask())
    assert seg.Modality == "SEG"
    assert seg.SOPClassUID == "1.2.840.10008.5.1.4.1.1.66.4"  # Segmentation Storage
    assert len(seg.SegmentSequence) == 3
    labels = [s.SegmentLabel for s in seg.SegmentSequence]
    assert labels == list(ONCOSEG_SEGMENT_LABELS)


def test_seg_references_source_study(series):
    seg = mask_to_dicom_seg(series, _three_channel_mask())
    # The SEG must live in the same study and frame of reference as its source.
    assert seg.StudyInstanceUID == series[0].StudyInstanceUID
    assert seg.FrameOfReferenceUID == series[0].FrameOfReferenceUID
    assert seg.SeriesInstanceUID != series[0].SeriesInstanceUID  # but a NEW series


def test_round_trips_to_disk(series, tmp_path):
    seg = mask_to_dicom_seg(series, _three_channel_mask())
    out = tmp_path / "oncoseg_seg.dcm"
    seg.save_as(out)
    reloaded = hd.seg.segread(out)
    assert len(reloaded.SegmentSequence) == 3
    # Pixels survive the round trip for the Whole Tumor segment.
    arr = reloaded.get_pixels_by_source_instance(
        source_sop_instance_uids=[img.SOPInstanceUID for img in series],
        segment_numbers=[2],
    )
    assert arr.sum() > 0  # Whole Tumor pixels survive the round trip


def test_single_segment_mask_3d(series):
    mask = np.zeros((3, 8, 8), dtype=np.uint8)
    mask[:, 3:5, 3:5] = 1
    seg = mask_to_dicom_seg(series, mask, segment_labels=["Whole Tumor"])
    assert len(seg.SegmentSequence) == 1


def test_segment_label_count_mismatch_raises(series):
    with pytest.raises(ValueError, match="label"):
        mask_to_dicom_seg(series, _three_channel_mask(), segment_labels=["only-one"])


def test_frame_count_mismatch_raises(series):
    bad = np.zeros((5, 8, 8, 3), dtype=np.uint8)  # 5 frames vs 3 source slices
    with pytest.raises(ValueError, match="frame"):
        mask_to_dicom_seg(series, bad)


def test_algorithm_metadata_recorded(series):
    seg = mask_to_dicom_seg(series, _three_channel_mask(), algorithm_version="2.3.4")
    assert seg.Manufacturer == "OncoSeg"
    assert "2.3.4" in str(seg.SoftwareVersions)
