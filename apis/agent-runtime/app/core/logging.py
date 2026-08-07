import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


def configure_logging(
    level: str = "INFO",
    service_name: str = "agent-runtime",
    log_dir: str = "/app/logs",
) -> None:
    """Configure logging to both stdout and a unique per-start file."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    try:
        logs_path = Path(log_dir)
        logs_path.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_file = logs_path / f"{service_name}-{started_at}.log"
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    except OSError:
        pass

    for handler in handlers:
        handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    for handler in handlers:
        root.addHandler(handler)

    # Route uvicorn logs through root handlers to persist access/error logs too.
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.setLevel(log_level)
        logger.propagate = True

    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("psycopg").setLevel(logging.WARNING)