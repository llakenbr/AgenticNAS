"""Database helpers for model selection and model spec loading.

This module is safe to import in notebooks: it does not run queries at import
time, and it lazily creates configuration and SQLAlchemy engines when needed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def find_repo_root(start_path: Path | None = None) -> Path:
    current_path = (start_path or Path.cwd()).resolve()
    for candidate in [current_path, *current_path.parents]:
        if (candidate / "apis").exists() and (candidate / "n8n").exists():
            return candidate
    raise RuntimeError("Could not locate repo root from the current working directory.")


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cleaned_value = value.strip().strip('"').strip("'")
        values[key.strip()] = cleaned_value
    return values


@dataclass(frozen=True, slots=True)
class Config:
    repo_root: Path
    n8n_env_path: Path
    n8n_env: dict[str, str]
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    trainer_api_base: str
    request_timeout: float
    database_url: str


@lru_cache(maxsize=1)
def get_config() -> Config:
    repo_root = find_repo_root()
    n8n_env_path = repo_root / "n8n" / ".env"
    n8n_env = read_env_file(n8n_env_path)

    db_host = os.getenv("MODEL_DB_HOST", "localhost")
    db_port = int(os.getenv("MODEL_DB_PORT", "5432"))
    db_name = os.getenv("MODEL_DB_NAME", n8n_env.get("MODEL_DB_NAME", "model_db"))
    db_user = os.getenv("MODEL_DB_USER", n8n_env.get("MODEL_DB_USER", "model_db_user"))
    db_password = os.getenv("MODEL_DB_PASSWORD", n8n_env.get("MODEL_DB_PASSWORD", ""))

    trainer_api_base = os.getenv("TRAINER_API_BASE", "http://localhost:7007")
    request_timeout = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    database_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    return Config(
        repo_root=repo_root,
        n8n_env_path=n8n_env_path,
        n8n_env=n8n_env,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        trainer_api_base=trainer_api_base,
        request_timeout=request_timeout,
        database_url=database_url,
    )


def get_engine(config: Config | None = None) -> Engine:
    active_config = config or get_config()
    return create_engine(active_config.database_url, pool_pre_ping=True)


def _load_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def list_models(limit: int = 20, config: Config | None = None) -> pd.DataFrame:
    query = text(
        """
        SELECT id, name, created_at, updated_at
        FROM public.models
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )
    with get_engine(config).connect() as conn:
        return pd.read_sql(query, conn, params={"limit": limit})

def get_session_models(session_id: str,config: Config | None = None) -> pd.DataFrame:
    query = text(
        """
        SELECT id, name, nn_taxonomy, created_at, updated_at
        FROM public.models
        WHERE session_id = :session_id
        ORDER BY created_at ASC
        """
    )
    with get_engine(config).connect() as conn:
        df = pd.read_sql(query, conn, params={"session_id": session_id})
    if df.empty:
        return None
    return df

def get_session_training_progress(session_id: str, config: Config | None = None) -> pd.DataFrame:
    query = text(
        """
        WITH ranked AS (
            SELECT
                model_id,
                created_at,
                COALESCE(
                    NULLIF(metrics#>>'{test_accuracy,raw}', '')::double precision,
                    NULLIF(metrics#>>'{accuracy,raw}', '')::double precision,
                    NULLIF(metrics#>>'{val_acc,raw}', '')::double precision,
                    NULLIF(metrics#>>'{val_accuracy,raw}', '')::double precision,
                    NULLIF(metrics->>'test_accuracy', '')::double precision,
                    NULLIF(metrics->>'accuracy', '')::double precision,
                    NULLIF(metrics->>'val_acc', '')::double precision,
                    NULLIF(metrics->>'val_accuracy', '')::double precision
                ) AS accuracy_score,
                COALESCE(
                    NULLIF(metrics#>>'{test_loss,raw}', '')::double precision,
                    NULLIF(metrics->>'test_loss', '')::double precision
                ) AS test_loss_raw,
                metrics
            FROM public.model_benchmarks
            WHERE session_id = :session_id
            AND benchmark_name = 'training'
        )
        SELECT model_id, accuracy_score, test_loss_raw, created_at, metrics
        FROM ranked
        ORDER BY created_at ASC;
        """
    )
    with get_engine(config).connect() as conn:
        df = pd.read_sql(query, conn, params={"session_id": session_id})
    if df.empty:
        return None
    return df

def get_top_session_model(session_id: str, config: Config | None = None) -> dict[str, Any] | None:
    query = text(
        """
        WITH ranked AS (
            SELECT
                model_id,
                COALESCE(
                    CASE
                        WHEN jsonb_typeof(metrics->'test_accuracy') = 'number'
                            THEN (metrics->>'test_accuracy')::double precision
                        WHEN jsonb_typeof(metrics->'test_accuracy') = 'object'
                             AND NULLIF(metrics#>>'{test_accuracy,raw}', '') ~ '^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$'
                            THEN (metrics#>>'{test_accuracy,raw}')::double precision
                        ELSE NULL
                    END,
                    CASE
                        WHEN jsonb_typeof(metrics->'accuracy') = 'number'
                            THEN (metrics->>'accuracy')::double precision
                        WHEN NULLIF(metrics->>'accuracy', '') ~ '^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$'
                            THEN (metrics->>'accuracy')::double precision
                        ELSE NULL
                    END,
                    CASE
                        WHEN jsonb_typeof(metrics->'val_acc') = 'number'
                            THEN (metrics->>'val_acc')::double precision
                        WHEN NULLIF(metrics->>'val_acc', '') ~ '^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$'
                            THEN (metrics->>'val_acc')::double precision
                        ELSE NULL
                    END,
                    CASE
                        WHEN jsonb_typeof(metrics->'val_accuracy') = 'number'
                            THEN (metrics->>'val_accuracy')::double precision
                        WHEN NULLIF(metrics->>'val_accuracy', '') ~ '^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$'
                            THEN (metrics->>'val_accuracy')::double precision
                        ELSE NULL
                    END,
                    CASE
                        WHEN jsonb_typeof(metrics->'test_accuracy') = 'number'
                            THEN (metrics->>'test_accuracy')::double precision
                        WHEN NULLIF(metrics->>'test_accuracy', '') ~ '^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$'
                            THEN (metrics->>'test_accuracy')::double precision
                        ELSE NULL
                    END
                ) AS accuracy_score,
                COALESCE(
                    CASE
                        WHEN jsonb_typeof(metrics->'test_loss') = 'number'
                            THEN (metrics->>'test_loss')::double precision
                        WHEN jsonb_typeof(metrics->'test_loss') = 'object'
                             AND NULLIF(metrics#>>'{test_loss,raw}', '') ~ '^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$'
                            THEN (metrics#>>'{test_loss,raw}')::double precision
                        WHEN NULLIF(metrics->>'test_loss', '') ~ '^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$'
                            THEN (metrics->>'test_loss')::double precision
                        ELSE NULL
                    END
                ) AS test_loss_raw
            FROM public.model_benchmarks
            WHERE session_id = :session_id
              AND benchmark_name = 'training'
        )
        SELECT model_id, accuracy_score, test_loss_raw
        FROM ranked
        WHERE accuracy_score IS NOT NULL
        ORDER BY accuracy_score DESC, model_id ASC
        LIMIT 1
        """
    )
    with get_engine(config).connect() as conn:
        df = pd.read_sql(query, conn, params={"session_id": session_id})
    if df.empty:
        return None

    row = df.iloc[0]
    return {
        "model_id": str(row["model_id"]),
        "test_accuracy_raw": float(row["accuracy_score"]),
        "test_loss_raw": float(row["test_loss_raw"]) if pd.notna(row["test_loss_raw"]) else None,
    }


def load_model_spec(model_id: str, config: Config | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    query = text(
        """
        SELECT id, name, nn_taxonomy, extra_meta, created_at
        FROM public.models
        WHERE id = :model_id
        LIMIT 1
        """
    )
    with get_engine(config).connect() as conn:
        df = pd.read_sql(query, conn, params={"model_id": model_id})

    if df.empty:
        raise ValueError(f"No model found for id={model_id}")

    row = df.iloc[0].to_dict()
    model_spec = _load_json_value(row["nn_taxonomy"])
    if model_spec is None:
        raise ValueError(f"Model id={model_id} does not have a saved architecture spec")

    model_info = {
        "id": str(row["id"]),
        "name": row.get("name"),
        "created_at": str(row.get("created_at")),
        "extra_meta": _load_json_value(row.get("extra_meta")),
    }
    return model_info, model_spec


__all__ = [
    "Config",
    "find_repo_root",
    "get_config",
    "get_engine",
    "get_top_session_model",
    "list_models",
    "load_model_spec",
    "read_env_file",
]
