import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

from core.logging import configure_logging
from core.settings import settings
from db.models import SessionEpisode, SessionState
from db.session import Base, engine
from routes.memory import router as memory_router

configure_logging(
    level=settings.log_level if hasattr(settings, "log_level") else "INFO",
    service_name=settings.app_name,
    log_dir=settings.log_dir,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        Base.metadata.create_all(bind=engine)
    except OperationalError as exc:
        logger.error("agent-memory database unavailable: %s", exc)
        raise

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


app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
)


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.app_env,
            "tables": [SessionState.__tablename__, SessionEpisode.__tablename__],
        }
    )


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


app.include_router(memory_router)
