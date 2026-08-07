import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from core.logging import configure_logging
from core.settings import settings
from models.request import trainRequest
from models.response import (
    TrainJobCancelResponse,
    TrainJobResultPendingResponse,
    TrainJobResultResponse,
    TrainJobStatusResponse,
    TrainJobSubmitResponse,
)
from services.test_model import test_model
from services.train_model import train_model


configure_logging(
    level=settings.log_level if hasattr(settings, "log_level") else "INFO",
    service_name=settings.app_name,
    log_dir=settings.log_dir,
)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_executor = ThreadPoolExecutor(max_workers=1)
_jobs_lock = Lock()
_jobs: dict[str, dict[str, Any]] = {}
_active_job_id: str | None = None
_active_future: Future | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    logger.info(
        "Started %s v%s with device: %s",
        settings.app_name,
        settings.app_version,
        device,
    )
    yield
    # shutdown
    _executor.shutdown(wait=False, cancel_futures=False)


app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_model_spec(raw_model_spec: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(raw_model_spec, dict):
        return raw_model_spec
    try:
        parsed = json.loads(raw_model_spec)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model_spec is a string but not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("model_spec JSON must decode to an object")
    return parsed


def _run_training_job(job_id: str, model_spec: dict[str, Any], training_parameters: Any) -> None:
    global _active_job_id
    global _active_future

    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = _utc_now_iso()

    try:
        trained_model = train_model(
            model_spec,
            training_parameters=training_parameters,
            torch_device=device,
        )
        test_result = test_model(
            trained_model.torch_model,
            training_parameters,
            trained_model.test_dataloader,
            trained_model.torch_device,
        )

        result = {
            "accuracy": test_result["accuracy"],
            "test_loss": test_result["test_loss"],
            "train_loss": trained_model.train_loss,
            "epochs": training_parameters.epochs,
            "device": str(trained_model.torch_device),
        }

        with _jobs_lock:
            _jobs[job_id]["status"] = "succeeded"
            _jobs[job_id]["completed_at"] = _utc_now_iso()
            _jobs[job_id]["result"] = result
            _jobs[job_id]["error"] = None

        logger.info("Training job %s completed successfully", job_id)
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["completed_at"] = _utc_now_iso()
            _jobs[job_id]["error"] = str(exc)

        logger.exception("Training job %s failed", job_id)
    finally:
        with _jobs_lock:
            _active_job_id = None
            _active_future = None


def _submit_train_job(template: trainRequest) -> dict[str, Any]:
    global _active_job_id
    global _active_future

    model_spec = _parse_model_spec(template.model_spec)
    training_parameters = template.training_parameters

    with _jobs_lock:
        if _active_job_id is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "A training job is already running",
                    "active_job_id": _active_job_id,
                },
            )

        job_id = str(uuid4())
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": _utc_now_iso(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "result": None,
            "request": {
                "training_parameters": training_parameters.model_dump(),
                "model_spec_keys": sorted(model_spec.keys()),
            },
        }

        _active_job_id = job_id
        _active_future = _executor.submit(
            _run_training_job,
            job_id,
            model_spec,
            training_parameters,
        )

    logger.info(
        "Queued training job %s for dataset=%s, epochs=%s",
        job_id,
        training_parameters.dataset_name,
        training_parameters.epochs,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/train-jobs/{job_id}",
        "result_url": f"/train-jobs/{job_id}/result",
    }


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": settings.app_name,
            "version": settings.app_version,
            "dataset_options": settings.dataset_options,
        }
    )


@app.post("/submit-train-job", response_model=TrainJobSubmitResponse, status_code=202)
async def submit_train_job(template: trainRequest) -> TrainJobSubmitResponse:
    try:
        payload = _submit_train_job(template)
        logger.info("Created training job %s successfully", payload["job_id"])
        return TrainJobSubmitResponse.model_validate(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/train-jobs/{job_id}", response_model=TrainJobStatusResponse)
def get_train_job_status(job_id: str) -> TrainJobStatusResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    response = {
        "job_id": job["job_id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "error": job["error"],
        "result_ready": job["status"] == "succeeded",
    }
    return TrainJobStatusResponse.model_validate(response)


@app.get(
    "/train-jobs/{job_id}/result",
    response_model=TrainJobResultResponse | TrainJobResultPendingResponse,
)
def get_train_job_result(job_id: str) -> JSONResponse | TrainJobResultResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job["status"] in {"queued", "running"}:
        pending_response = TrainJobResultPendingResponse(
            job_id=job_id,
            status=job["status"],
            message="Training still in progress",
        )
        return JSONResponse(
            status_code=202,
            content=jsonable_encoder(pending_response),
        )

    if job["status"] == "failed":
        raise HTTPException(
            status_code=422,
            detail={
                "job_id": job_id,
                "status": "failed",
                "error": job["error"],
            },
        )

    return TrainJobResultResponse.model_validate(job["result"])


@app.delete("/train-jobs/{job_id}", response_model=TrainJobCancelResponse)
def cancel_train_job(job_id: str) -> TrainJobCancelResponse:
    global _active_job_id
    global _active_future

    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        if job["status"] not in {"queued", "running"}:
            return TrainJobCancelResponse(
                job_id=job_id,
                status=job["status"],
                message="Job is already completed",
            )

        if _active_job_id != job_id or _active_future is None:
            raise HTTPException(status_code=409, detail="Job is not active")

        cancelled = _active_future.cancel()
        if cancelled:
            job["status"] = "cancelled"
            job["completed_at"] = _utc_now_iso()
            _active_job_id = None
            _active_future = None
            return TrainJobCancelResponse(
                job_id=job_id,
                status="cancelled",
                message="Job cancelled before execution started",
            )

    raise HTTPException(
        status_code=409,
        detail={
            "job_id": job_id,
            "status": "running",
            "message": "Running jobs cannot be force-cancelled safely",
        },
    )


@app.get("/get_dataset_options")
def get_dataset_options():
    logger.info("Received request for dataset options")
    return JSONResponse(content=jsonable_encoder(settings.dataset_options))