"""FastAPI application factory for the OncoSeg inference service."""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from src.api.schemas import (
    ChannelStats,
    HealthResponse,
    InfoResponse,
    LesionMeasurement,
    MeasureResponse,
    ReadyResponse,
    RECISTReport,
    ResponseAssessment,
)
from src.api.service import (
    MODALITIES,
    OncoSegService,
    ServiceMeta,
    build_predictor_from_checkpoint,
)

logger = logging.getLogger(__name__)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_build_service_from_env() -> OncoSegService | None:
    """Construct a service from environment variables, or return None.

    Env vars:
        ONCOSEG_CHECKPOINT — path to .pth. Required.
        ONCOSEG_MODEL_SOURCE — "train_all" (default) | "src".
        ONCOSEG_ROI_SIZE — "H,W,D" (default "128,128,128").
        ONCOSEG_SW_BATCH_SIZE — int (default 2).
        ONCOSEG_MC_SAMPLES — int (default 0).
    """
    ckpt_env = os.environ.get("ONCOSEG_CHECKPOINT")
    if not ckpt_env:
        return None

    ckpt_path = Path(ckpt_env)
    if not ckpt_path.exists():
        logger.warning("ONCOSEG_CHECKPOINT=%s does not exist; service will not load", ckpt_path)
        return None

    roi_raw = os.environ.get("ONCOSEG_ROI_SIZE", "128,128,128")
    roi = tuple(int(x) for x in roi_raw.split(","))
    if len(roi) != 3:
        raise ValueError(f"ONCOSEG_ROI_SIZE must have 3 ints, got {roi_raw!r}")

    model_source = os.environ.get("ONCOSEG_MODEL_SOURCE", "train_all")
    sw_batch_size = int(os.environ.get("ONCOSEG_SW_BATCH_SIZE", "2"))
    mc_samples = int(os.environ.get("ONCOSEG_MC_SAMPLES", "0"))

    device = _select_device()
    predictor, name = build_predictor_from_checkpoint(
        checkpoint_path=ckpt_path,
        device=device,
        roi_size=roi,  # type: ignore[arg-type]
        sw_batch_size=sw_batch_size,
        mc_samples=mc_samples,
        model_source=model_source,
    )

    meta = ServiceMeta(
        model_name=name,
        model_source=model_source,
        checkpoint=str(ckpt_path),
        device=str(device),
        roi_size=roi,  # type: ignore[arg-type]
        num_classes=3,
        mc_samples=mc_samples,
    )
    return OncoSegService(predictor=predictor, meta=meta)


def get_service() -> OncoSegService:
    """FastAPI dependency — replaced per-app by `create_app`.

    The default implementation raises 503 so that importing the dependency
    symbol outside of a bound app still fails loudly.
    """
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Model not loaded. Set ONCOSEG_CHECKPOINT and restart.",
    )


ServiceDep = Annotated[OncoSegService, Depends(get_service)]


async def _collect_modalities(
    files: dict[str, UploadFile],
) -> dict[str, bytes]:
    blobs: dict[str, bytes] = {}
    for mod, upload in files.items():
        if upload is None:
            continue
        data = await upload.read()
        if not data:
            raise HTTPException(
                status_code=400, detail=f"Uploaded file for '{mod}' is empty"
            )
        blobs[mod] = data
    missing = [m for m in MODALITIES if m not in blobs]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required modalities: {missing}. Expected fields: {list(MODALITIES)}",
        )
    return blobs


def create_app(service: OncoSegService | None = None) -> FastAPI:
    """Build the FastAPI app.

    If `service` is provided it is used directly (useful for tests).
    Otherwise the app tries to construct one from env vars at startup.
    """
    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        if fastapi_app.state.service is None:
            try:
                fastapi_app.state.service = _maybe_build_service_from_env()
            except Exception:
                logger.exception("Failed to construct OncoSeg service from env")
                fastapi_app.state.service = None
        yield

    app = FastAPI(
        title="OncoSeg Inference API",
        description=(
            "3D tumor segmentation + automated RECIST response assessment.\n\n"
            "Upload 4 BraTS-format NIfTI modalities (t1n, t1c, t2w, t2f) and receive "
            "a segmentation mask or RECIST measurements."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.service = service

    def _resolve_service() -> OncoSegService:
        if app.state.service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Set ONCOSEG_CHECKPOINT and restart.",
            )
        return app.state.service

    app.dependency_overrides[get_service] = _resolve_service

    @app.get("/healthz", response_model=HealthResponse, tags=["health"])
    def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=ReadyResponse, tags=["health"])
    def readyz() -> ReadyResponse:
        svc = app.state.service
        if svc is None:
            return ReadyResponse(status="loading", model_loaded=False, checkpoint=None)
        return ReadyResponse(
            status="ready",
            model_loaded=True,
            checkpoint=svc.meta.checkpoint,
        )

    @app.get("/info", response_model=InfoResponse, tags=["health"])
    def info(svc: ServiceDep) -> InfoResponse:
        m = svc.meta
        return InfoResponse(
            model_name=m.model_name,
            model_source=m.model_source,
            checkpoint=m.checkpoint,
            device=m.device,
            roi_size=m.roi_size,
            num_classes=m.num_classes,
            mc_samples=m.mc_samples,
        )

    @app.post(
        "/predict/measure",
        response_model=MeasureResponse,
        tags=["predict"],
        summary="Segment + compute RECIST metrics",
    )
    async def predict_measure(
        svc: ServiceDep,
        t1n: Annotated[UploadFile, File(description="Native T1 NIfTI")],
        t1c: Annotated[UploadFile, File(description="Post-contrast T1 NIfTI")],
        t2w: Annotated[UploadFile, File(description="T2-weighted NIfTI")],
        t2f: Annotated[UploadFile, File(description="T2-FLAIR NIfTI")],
        subject_id: Annotated[str, Form()] = "subject",
    ) -> MeasureResponse:
        blobs = await _collect_modalities({"t1n": t1n, "t1c": t1c, "t2w": t2w, "t2f": t2f})
        try:
            result = svc.measure(blobs, subject_id=subject_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return MeasureResponse(
            subject_id=result["subject_id"],
            shape=result["shape"],
            pixdim=result["pixdim"],
            channel_stats=[ChannelStats(**cs) for cs in result["channel_stats"]],
            recist=RECISTReport(
                num_lesions=result["recist"]["num_lesions"],
                sum_longest_diameter_mm=result["recist"]["sum_longest_diameter_mm"],
                total_volume_mm3=result["recist"]["total_volume_mm3"],
                lesions=[LesionMeasurement(**les) for les in result["recist"]["lesions"]],
            ),
        )

    @app.post(
        "/predict/segment",
        tags=["predict"],
        summary="Segment 4-modality MRI, return NIfTI mask",
        response_class=FileResponse,
    )
    async def predict_segment(
        svc: ServiceDep,
        t1n: Annotated[UploadFile, File()],
        t1c: Annotated[UploadFile, File()],
        t2w: Annotated[UploadFile, File()],
        t2f: Annotated[UploadFile, File()],
        subject_id: Annotated[str, Form()] = "subject",
    ) -> FileResponse:
        blobs = await _collect_modalities({"t1n": t1n, "t1c": t1c, "t2w": t2w, "t2f": t2f})
        try:
            segmentation, affine, _pixdim = svc.segment(blobs, subject_id=subject_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        tmp_dir = Path(tempfile.mkdtemp(prefix="oncoseg_api_"))
        out_path = tmp_dir / f"{subject_id}_segmentation.nii.gz"
        svc.write_segmentation_nifti(segmentation, affine, out_path)
        return FileResponse(
            path=str(out_path),
            media_type="application/gzip",
            filename=out_path.name,
        )

    @app.post(
        "/predict/response",
        response_model=ResponseAssessment,
        tags=["predict"],
        summary="Classify CR/PR/SD/PD from baseline + follow-up scans",
    )
    async def predict_response(
        svc: ServiceDep,
        baseline_t1n: Annotated[UploadFile, File()],
        baseline_t1c: Annotated[UploadFile, File()],
        baseline_t2w: Annotated[UploadFile, File()],
        baseline_t2f: Annotated[UploadFile, File()],
        followup_t1n: Annotated[UploadFile, File()],
        followup_t1c: Annotated[UploadFile, File()],
        followup_t2w: Annotated[UploadFile, File()],
        followup_t2f: Annotated[UploadFile, File()],
        subject_id: Annotated[str, Form()] = "subject",
    ) -> ResponseAssessment:
        baseline = await _collect_modalities({
            "t1n": baseline_t1n, "t1c": baseline_t1c,
            "t2w": baseline_t2w, "t2f": baseline_t2f,
        })
        followup = await _collect_modalities({
            "t1n": followup_t1n, "t1c": followup_t1c,
            "t2w": followup_t2w, "t2f": followup_t2f,
        })
        try:
            result = svc.response(baseline, followup, subject_id=subject_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ResponseAssessment(**result)

    return app


app = create_app()
