"""Trainer API helpers for benchmarking notebooks."""

from __future__ import annotations

import time
from typing import Any

import requests

from .db_con import Config, get_config


def _get_api_settings(config: Config | None = None) -> tuple[str, float]:
    active_config = config or get_config()
    return active_config.trainer_api_base, active_config.request_timeout


def submit_train_job(payload: dict[str, Any], config: Config | None = None) -> dict[str, Any]:
    trainer_api_base, request_timeout = _get_api_settings(config)
    url = f"{trainer_api_base}/submit-train-job"
    response = requests.post(url, json=payload, timeout=request_timeout)
    response.raise_for_status()
    return response.json()


def get_job_status(job_id: str, config: Config | None = None) -> dict[str, Any]:
    trainer_api_base, request_timeout = _get_api_settings(config)
    url = f"{trainer_api_base}/train-jobs/{job_id}"
    response = requests.get(url, timeout=request_timeout)
    response.raise_for_status()
    return response.json()


def get_job_result(job_id: str, config: Config | None = None) -> dict[str, Any]:
    trainer_api_base, request_timeout = _get_api_settings(config)
    url = f"{trainer_api_base}/train-jobs/{job_id}/result"
    response = requests.get(url, timeout=request_timeout)
    response.raise_for_status()
    return response.json()


def wait_for_job(
    job_id: str,
    timeout_seconds: float = 3600,
    poll_interval_seconds: float = 60,
    config: Config | None = None,
) -> dict[str, Any]:
    started = time.time()
    while True:
        status = get_job_status(job_id, config=config)
        state = status.get("status")
        print(f"job={job_id} status={state}")

        if state in {"succeeded", "failed", "cancelled"}:
            return status

        if (time.time() - started) > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for job {job_id}")

        time.sleep(poll_interval_seconds)


__all__ = [
    "get_job_result",
    "get_job_status",
    "submit_train_job",
    "wait_for_job",
]
