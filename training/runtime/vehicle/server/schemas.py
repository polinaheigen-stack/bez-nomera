from typing import Literal
from pydantic import BaseModel, Field, model_validator

Mode = Literal['prod']
Device = Literal['cpu', 'cuda']


class DeviceChoice(BaseModel):
    device: Device


class DeviceOption(BaseModel):
    id: Device
    name: str
    available: bool
    reason: str | None


class RuntimeSettings(BaseModel):
    selected_device: Device
    active_device: Device | None
    switching: bool
    can_switch: bool
    devices: list[DeviceOption]
    error: str | None


class ModelInfo(BaseModel):
    name: str
    version: str
    sha256: str | None
    dimension: int | None
    device: str
    inference_fingerprint: str | None = None
    retrieval_fingerprint: str | None = None


class Limits(BaseModel):
    max_images: int = 2000
    max_file_mb: int = 32
    max_upload_mb: int = 2048
    max_pending_runs: int = 8


class ServiceStatus(BaseModel):
    contract_version: Literal['1'] = '1'
    mode: Mode
    available: bool
    reason: str | None
    model: ModelInfo | None
    threshold: float | None
    limits: Limits


class Gallery(BaseModel):
    id: str
    name: str
    count: int
    processed: int
    status: Literal['indexing', 'ready', 'failed']
    created_at: str
    error: str | None


class ImageRef(BaseModel):
    image_id: str
    image_url: str
    crop_url: str
    bbox: tuple[int, int, int, int]
    width: int
    height: int


class Candidate(BaseModel):
    rank: int
    image: ImageRef
    score: float
    accepted: bool


class QueryResult(BaseModel):
    query: ImageRef
    status: Literal['matched', 'rejected', 'error']
    candidates: list[Candidate]
    duration_ms: float
    error: str | None


class RunTiming(BaseModel):
    queue_ms: float = Field(ge=0)
    processing_ms: float = Field(ge=0)
    successful_images: int = Field(ge=0)
    images_per_second: float | None = Field(default=None, ge=0)
    mean_image_ms: float | None = Field(default=None, ge=0)


class Run(BaseModel):
    evaluation: 'RunEvaluation' = Field(default_factory=lambda: RunEvaluation())
    timing: RunTiming | None = None
    id: str
    mode: Mode
    status: Literal['queued', 'running', 'completed', 'failed', 'cancelled']
    gallery_id: str
    model: ModelInfo | None
    threshold: float
    total: int
    processed: int
    failed: int
    created_at: str
    results: list[QueryResult]
    error: str | None
    export_ready: bool
    artifacts_url: str | None
    evidence_url: str


class Metrics(BaseModel):
    map_at_10: float | None = None
    rank_1: float | None = None
    rank_5: float | None = None
    micro_f1: float | None = None
    tnr: float | None = None
    latency_ms: float | None = None
    throughput_fps: float | None = None
    peak_ram_mb: float | None = None
    peak_vram_mb: float | None = None


class EvaluationReport(BaseModel):
    mode: Mode
    measured: bool = False
    model: ModelInfo | None
    dataset: str | None = None
    hardware: str | None = None
    run_id: str | None = None
    calibration_sha256: str | None = None
    ground_truth_sha256: str | None = None
    metrics: Metrics | None = None
    notes: list[str]

    @model_validator(mode='after')
    def measured_metrics_only(self):
        if self.measured:
            if self.metrics is None or not any(value is not None for value in self.metrics.model_dump().values()):
                raise ValueError('Measured report must contain actual metrics.')
        elif self.metrics is not None:
            raise ValueError('Unmeasured report must have metrics=null.')
        return self


class RunEvaluation(BaseModel):
    status: Literal['unavailable', 'waiting', 'running', 'completed', 'failed'] = 'unavailable'
    report: EvaluationReport | None = None
    error: str | None = None
    ground_truth_sha256: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ApiError(BaseModel):
    error: ErrorDetail


class Evidence(BaseModel):
    timing: RunTiming | None = None
    contract_version: Literal['1'] = '1'
    mode: Mode
    run_id: str
    status: str
    model: ModelInfo | None
    calibration_sha256: str | None
    threshold: float
    score_definition: str
    retrieval: dict | None = None
    retrieval_fingerprint: str | None = None
    gallery_id: str
    gallery_sha256: str
    input_order: dict[str, list[str]]
    inputs: dict[str, list[dict]]
    files_sha256: dict[str, str]
    events: list[dict]
    notes: list[str]
