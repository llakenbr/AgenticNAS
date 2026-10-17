from typing import Literal

from pydantic import BaseModel


JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class TrainJobSubmitResponse(BaseModel):
    job_id: str
    status: Literal["queued"]
    status_url: str
    result_url: str


class TrainJobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    result_ready: bool


class TrainJobResultResponse(BaseModel):
    accuracy: float
    test_loss: float
    train_loss: list[float]
    epochs: int
    device: str


class TrainJobResultPendingResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running"]
    message: str


class TrainJobCancelResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
