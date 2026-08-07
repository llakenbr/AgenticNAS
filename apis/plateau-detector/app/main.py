import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select

from db.session import SessionLocal
from db.models import PlateauHistory

from core.logging import configure_logging
from core.settings import settings
from db.session import Base, engine

configure_logging(
    level=settings.log_level if hasattr(settings, "log_level") else "INFO",
    service_name=settings.app_name,
    log_dir=settings.log_dir,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    Base.metadata.create_all(bind=engine)
    logger.info(
        "Started %s v%s [%s] — postgres: %s:%s/%s",
        settings.app_name,
        settings.app_version,
        settings.app_env,
        settings.postgres_host,
        settings.postgres_port,
        settings.postgres_db,
    )
    yield
    # shutdown

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_observation_items(body: Any) -> list[dict[str, Any]]:
    if isinstance(body, list):
        return [item for item in body if isinstance(item, dict)]

    if not isinstance(body, dict):
        return []

    for key in ("observations", "results", "items", "updates", "outputs", "history", "records"):
        value = body.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    if isinstance(body.get("output"), dict):
        return [body]

    if "model_test_accuracy" in body or "model_ID" in body or "model_id" in body:
        return [body]

    return []


def _extract_output(item: dict[str, Any]) -> dict[str, Any]:
    output = item.get("output")
    if isinstance(output, dict):
        return output
    return item


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.app_env,
        }
    )


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/update")
async def update(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object")

    session_id = body.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise HTTPException(status_code=422, detail="session_id is required")
    session_id = session_id.strip()

    window_size = int(body.get("window_size", settings.window_size))
    min_gain = float(body.get("min_gain", settings.min_gain))

    observation_items = _extract_observation_items(body)
    if not observation_items:
        raise HTTPException(
            status_code=422,
            detail="Request must include at least one observation under output/results/items/updates/outputs/history/records",
        )

    inserted_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    db = SessionLocal()
    try:
        for item in observation_items:
            output = _extract_output(item)
            model_test_accuracy = _as_float(output.get("model_test_accuracy"))

            if model_test_accuracy is None:
                skipped_rows.append(item)
                continue

            entry = PlateauHistory(
                session_id=session_id,
                model_test_accuracy=model_test_accuracy,
                payload=item,
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)

            inserted_rows.append(
                {
                    "session_id": session_id,
                    "model_test_accuracy": model_test_accuracy,
                    "payload": item,
                }
            )

        stmt = (
            select(PlateauHistory.model_test_accuracy)
            .where(PlateauHistory.session_id == session_id)
            .order_by(PlateauHistory.created_at.desc(), PlateauHistory.id.desc())
            .limit(window_size)
        )
        rows = db.execute(stmt).scalars().all()
    finally:
        db.close()

    accuracies = [float(score) for score in rows if score is not None]
    valid_count = len(accuracies)
    best_in_window = max(accuracies) if accuracies else None
    gain = (max(accuracies) - min(accuracies)) if accuracies else None
    plateau = valid_count == window_size and gain is not None and gain < min_gain

    logger.info(
        "update result session_id=%s window_size=%s min_gain=%s valid_count=%s inserted_count=%s skipped_count=%s best_in_window=%s gain=%s plateau=%s",
        session_id,
        window_size,
        min_gain,
        valid_count,
        len(inserted_rows),
        len(skipped_rows),
        best_in_window,
        gain,
        plateau,
    )

    response = {
        "plateau": plateau,
        "valid_count": valid_count,
        "best_in_window": best_in_window,
        "gain": gain,
        "window_size": window_size,
        "min_gain": min_gain,
        "received_request": {
            "session_id": session_id,
            "window_size": window_size,
            "min_gain": min_gain,
            "observations": observation_items,
        },
        "inserted_rows": inserted_rows,
        "skipped_count": len(skipped_rows),
    }
    return JSONResponse(content=response)