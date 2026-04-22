"""Pydantic schemas for the OncoSeg REST API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])


class ReadyResponse(BaseModel):
    status: str = Field(..., examples=["ready", "loading"])
    model_loaded: bool
    checkpoint: str | None = None


class InfoResponse(BaseModel):
    model_name: str
    model_source: str = Field(..., description="train_all | src")
    checkpoint: str | None
    device: str
    roi_size: tuple[int, int, int]
    num_classes: int
    mc_samples: int
    in_channels: int = 4
    channel_order: list[str] = Field(
        default_factory=lambda: ["t1n", "t1c", "t2w", "t2f"]
    )
    output_channels: list[str] = Field(
        default_factory=lambda: ["TC", "WT", "ET"],
        description="Tumor Core / Whole Tumor / Enhancing Tumor",
    )


class LesionMeasurement(BaseModel):
    id: int
    longest_diameter_mm: float
    volume_mm3: float
    voxel_count: int


class RECISTReport(BaseModel):
    num_lesions: int
    sum_longest_diameter_mm: float
    total_volume_mm3: float
    lesions: list[LesionMeasurement]


class ChannelStats(BaseModel):
    name: str
    positive_voxels: int
    volume_mm3: float


class MeasureResponse(BaseModel):
    subject_id: str
    shape: tuple[int, int, int]
    pixdim: tuple[float, float, float]
    channel_stats: list[ChannelStats]
    recist: RECISTReport


class ResponseAssessment(BaseModel):
    category: str = Field(..., examples=["CR", "PR", "SD", "PD"])
    category_full: str
    percent_change: float
    baseline_sum_ld_mm: float
    followup_sum_ld_mm: float
    baseline_volume_mm3: float
    followup_volume_mm3: float
    num_baseline_lesions: int
    num_followup_lesions: int


class APIError(BaseModel):
    detail: str
