# app/main.py
import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from core.logging import configure_logging
from core.settings import settings
from db.session import Base, engine
from routes.agents import router as agents_router
from routes.health import router as health_router
from routes.runs import router as runs_router
from services.graph_factory import init_checkpointer

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
    init_checkpointer()
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


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.app_env,
        }
    )


app.include_router(health_router)
app.include_router(agents_router)
app.include_router(runs_router)