"""End-to-end tests for the OncoSeg FastAPI service.

Uses a deterministic FakePredictor so tests don't need a GPU or checkpoint.
Real NIfTI byte blobs are uploaded via TestClient to exercise the full
request -> preprocessing -> inference -> RECIST -> response path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.service import MODALITIES, OncoSegService, ServiceMeta


class _FakePredictor:
    """Predictor stub returning a caller-specified segmentation.

    `seg_factory` is called with the preprocessed input tensor; the returned
    [3, H, W, D] array becomes the segmentation output.
    """

    def __init__(self, seg_factory):
        self.seg_factory = seg_factory
        self.calls = 0

    def predict_volume(self, image: torch.Tensor) -> dict[str, np.ndarray]:
        self.calls += 1
        seg = self.seg_factory(image)
        probs = seg.astype(np.float32)
        return {"segmentation": seg, "probabilities": probs}


def _make_nifti_bytes(data: np.ndarray, affine: np.ndarray | None = None) -> bytes:
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        nib.save(img, tmp.name)
        tmp_path = Path(tmp.name)
    try:
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def _four_modalities(shape=(32, 32, 32)) -> dict[str, bytes]:
    rng = np.random.default_rng(seed=0)
    return {
        mod: _make_nifti_bytes(rng.normal(size=shape).astype(np.float32))
        for mod in MODALITIES
    }


def _service_with_seg(seg: np.ndarray, model_source: str = "fake") -> tuple[OncoSegService, _FakePredictor]:
    predictor = _FakePredictor(seg_factory=lambda image: seg)
    meta = ServiceMeta(
        model_name="Fake",
        model_source=model_source,
        checkpoint="/fake/path.pth",
        device="cpu",
        roi_size=(32, 32, 32),
        num_classes=3,
        mc_samples=0,
    )
    return OncoSegService(predictor=predictor, meta=meta), predictor


@pytest.fixture
def two_lesion_seg():
    """3-channel seg [TC, WT, ET] with two ET lesions for deterministic RECIST.

    Lesion sizes are chosen so each longest axial diameter clears the
    RECIST 1.1 §3.1.1 10mm target-lesion threshold at 1mm spacing.
    """
    # The test NIfTI uploads are 32^3, but the inference pipeline resamples to
    # a 1mm pixdim (no-op for 1mm inputs) -> the predict_volume input spatial
    # dims equal the post-transform input. Our FakePredictor ignores the input
    # and returns this fixed segmentation, so its shape defines the output shape.
    seg = np.zeros((3, 32, 32, 32), dtype=np.uint8)
    # ET lesion 1: 12x12x12 cube -> longest axial diameter 11*sqrt(2) ≈ 15.56mm
    seg[2, 4:16, 4:16, 4:16] = 1
    # ET lesion 2: 9x9x9 cube, disjoint -> longest axial diameter 8*sqrt(2) ≈ 11.31mm
    seg[2, 20:29, 20:29, 20:29] = 1
    # WT contains ET ∪ a slightly larger envelope around lesion 1
    seg[1] = seg[2].copy()
    seg[1, 2:18, 2:18, 2:18] = 1
    # TC: includes both lesions, tighter than WT around lesion 1
    seg[0, 4:16, 4:16, 4:16] = 1
    seg[0, 20:29, 20:29, 20:29] = 1
    return seg


@pytest.fixture
def client(two_lesion_seg):
    service, predictor = _service_with_seg(two_lesion_seg)
    app = create_app(service=service)
    client = TestClient(app)
    client.app.state._test_predictor = predictor
    return client


class TestHealth:
    def test_healthz(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_readyz_when_loaded(self, client):
        r = client.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["model_loaded"] is True
        assert body["checkpoint"] == "/fake/path.pth"

    def test_readyz_when_unloaded(self):
        app = create_app(service=None)
        # prevent the startup hook from populating from env in test
        app.state.service = None
        c = TestClient(app)
        r = c.get("/readyz")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is False

    def test_info(self, client):
        r = client.get("/info")
        assert r.status_code == 200
        body = r.json()
        assert body["model_name"] == "Fake"
        assert body["num_classes"] == 3
        assert body["channel_order"] == ["t1n", "t1c", "t2w", "t2f"]
        assert body["output_channels"] == ["TC", "WT", "ET"]

    def test_info_503_when_unloaded(self):
        app = create_app(service=None)
        app.state.service = None
        c = TestClient(app)
        r = c.get("/info")
        assert r.status_code == 503


class TestPredictMeasure:
    def test_happy_path(self, client):
        files = {
            mod: (f"{mod}.nii.gz", blob, "application/gzip")
            for mod, blob in _four_modalities().items()
        }
        r = client.post("/predict/measure", files=files, data={"subject_id": "patient_001"})
        assert r.status_code == 200, r.text

        body = r.json()
        assert body["subject_id"] == "patient_001"
        assert len(body["shape"]) == 3

        channel_names = [cs["name"] for cs in body["channel_stats"]]
        assert channel_names == ["TC", "WT", "ET"]

        recist = body["recist"]
        assert recist["num_lesions"] == 2
        # ET lesion 1 is a 4x4x4 cube at 1mm spacing => sqrt(2)*3 ≈ 4.24mm
        # ET lesion 2 is a 2x2x2 cube at 1mm spacing => sqrt(2)*1 ≈ 1.41mm
        # Predictor was called exactly once (no response classification)
        assert client.app.state._test_predictor.calls == 1

    def test_missing_modality_rejected(self, client):
        blobs = _four_modalities()
        files = {
            "t1n": ("t1n.nii.gz", blobs["t1n"], "application/gzip"),
            "t1c": ("t1c.nii.gz", blobs["t1c"], "application/gzip"),
            "t2w": ("t2w.nii.gz", blobs["t2w"], "application/gzip"),
            # no t2f
        }
        r = client.post("/predict/measure", files=files)
        # FastAPI validates the required File fields -> 422
        assert r.status_code == 422

    def test_empty_upload_rejected(self, client):
        blobs = _four_modalities()
        files = {
            "t1n": ("t1n.nii.gz", b"", "application/gzip"),
            "t1c": ("t1c.nii.gz", blobs["t1c"], "application/gzip"),
            "t2w": ("t2w.nii.gz", blobs["t2w"], "application/gzip"),
            "t2f": ("t2f.nii.gz", blobs["t2f"], "application/gzip"),
        }
        r = client.post("/predict/measure", files=files)
        assert r.status_code == 400
        assert "empty" in r.json()["detail"].lower()


class TestPredictSegment:
    def test_returns_nifti(self, client, tmp_path):
        files = {
            mod: (f"{mod}.nii.gz", blob, "application/gzip")
            for mod, blob in _four_modalities().items()
        }
        r = client.post("/predict/segment", files=files, data={"subject_id": "sub42"})
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "application/gzip"
        assert "sub42_segmentation.nii.gz" in r.headers["content-disposition"]

        # Round-trip: load the returned NIfTI and check it's 4D with 3 channels
        out = tmp_path / "returned.nii.gz"
        out.write_bytes(r.content)
        img = nib.load(out)
        arr = img.get_fdata()
        assert arr.ndim == 4
        assert arr.shape[-1] == 3  # channels last per write_segmentation_nifti


class TestPredictResponse:
    def test_stable_disease_when_masks_identical(self, client):
        baseline_blobs = _four_modalities()
        followup_blobs = _four_modalities()
        files = {}
        for mod, blob in baseline_blobs.items():
            files[f"baseline_{mod}"] = (f"b_{mod}.nii.gz", blob, "application/gzip")
        for mod, blob in followup_blobs.items():
            files[f"followup_{mod}"] = (f"f_{mod}.nii.gz", blob, "application/gzip")

        r = client.post("/predict/response", files=files, data={"subject_id": "pt01"})
        assert r.status_code == 200, r.text
        body = r.json()
        # FakePredictor returns the same segmentation for both timepoints
        # => no diameter change => Stable Disease
        assert body["category"] == "SD"
        assert body["percent_change"] == 0.0
        assert body["num_baseline_lesions"] == 2
        assert body["num_followup_lesions"] == 2
        # Predictor called twice: once per timepoint
        assert client.app.state._test_predictor.calls == 2

    def test_complete_response_when_followup_empty(self, two_lesion_seg):
        # Baseline returns two-lesion seg; followup returns all-zero seg.
        empty_seg = np.zeros_like(two_lesion_seg)
        calls = {"n": 0}

        def seg_factory(_image):
            calls["n"] += 1
            return two_lesion_seg if calls["n"] == 1 else empty_seg

        service = OncoSegService(
            predictor=_FakePredictor(seg_factory=seg_factory),
            meta=ServiceMeta(
                model_name="Fake", model_source="fake", checkpoint=None,
                device="cpu", roi_size=(32, 32, 32), num_classes=3, mc_samples=0,
            ),
        )
        app = create_app(service=service)
        c = TestClient(app)

        blobs_b = _four_modalities()
        blobs_f = _four_modalities()
        files = {}
        for mod, blob in blobs_b.items():
            files[f"baseline_{mod}"] = (f"b_{mod}.nii.gz", blob, "application/gzip")
        for mod, blob in blobs_f.items():
            files[f"followup_{mod}"] = (f"f_{mod}.nii.gz", blob, "application/gzip")

        r = c.post("/predict/response", files=files)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["category"] == "CR"
        assert body["num_followup_lesions"] == 0


class TestBuildPredictorFromCheckpoint:
    def test_rejects_unknown_model_source(self, tmp_path: Path):
        from src.api.service import build_predictor_from_checkpoint

        # Any .pth file — we'll never read it because source validation runs first
        ckpt = tmp_path / "dummy.pth"
        torch.save({"model_state_dict": {}}, ckpt)

        with pytest.raises(ValueError, match="Unknown model_source"):
            build_predictor_from_checkpoint(
                checkpoint_path=ckpt,
                device=torch.device("cpu"),
                model_source="not_a_real_source",
            )


class TestPredictMeasureDicom:
    def test_dicom_zip_round_trip(self, client):
        """Four DICOM-series zips (one per modality) should route through
        /predict/measure/dicom and produce the same RECIST response shape
        as the NIfTI endpoint."""
        import io
        import zipfile

        from tests.test_dicom import _write_series

        def _make_zip(series_dir: Path) -> bytes:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                for path in series_dir.iterdir():
                    zf.write(path, arcname=path.name)
            return buf.getvalue()

        # Small synthetic volume — FakePredictor ignores input shape/content
        # and returns the fixture's two-lesion segmentation unchanged.
        vol = np.zeros((16, 16, 8), dtype=np.int16)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            zips: dict[str, bytes] = {}
            for mod in MODALITIES:
                series_dir = tmp_root / mod
                _write_series(series_dir, vol)
                zips[mod] = _make_zip(series_dir)

        files = {mod: (f"{mod}.zip", blob, "application/zip") for mod, blob in zips.items()}
        r = client.post("/predict/measure/dicom", files=files, data={"subject_id": "dcm_01"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["subject_id"] == "dcm_01"
        assert body["recist"]["num_lesions"] == 2  # from two_lesion_seg fixture

    def test_dicom_endpoint_rejects_bad_zip(self, client):
        """A non-DICOM zip should surface as 400 with a load error."""
        import io
        import zipfile

        def _garbage_zip() -> bytes:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("not_a_dicom.txt", "hello world")
            return buf.getvalue()

        files = {
            mod: (f"{mod}.zip", _garbage_zip(), "application/zip")
            for mod in MODALITIES
        }
        r = client.post("/predict/measure/dicom", files=files)
        assert r.status_code == 400
        assert "No DICOM" in r.json()["detail"]


class TestInferEmbedDim:
    def test_detects_from_patch_embed(self):
        from src.api.service import _infer_embed_dim

        # OncoSeg patch_embed.proj is Conv3d(in=4, out=embed_dim, k=4)
        sd = {"encoder.patch_embed.proj.weight": torch.zeros(24, 4, 4, 4, 4)}
        assert _infer_embed_dim(sd) == 24

    def test_returns_none_when_absent(self):
        from src.api.service import _infer_embed_dim

        assert _infer_embed_dim({}) is None
