# ./app/routes/runs.py
import logging
from typing import Iterable

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.models import RegisteredAgent
from db.session import get_session
from models.run import RunRequest, RunResponse
from services.agent_executor import invoke_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents/{agent_id}/sessions/{session_id}", tags=["runs"])


def _exception_chain(exc: BaseException) -> Iterable[BaseException]:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = " ".join(str(item) for item in _exception_chain(exc)).lower()
    return any(
        marker in text
        for marker in (
            "ratelimitreached",
            "rate limit exceeded",
            "too many requests",
            "quota request exceeds the tokens limit",
            "error code: 429",
            "status code: 429",
        )
    )


@router.post("/runs", response_model=RunResponse)
def run_agent(
    agent_id: str,
    session_id: str,
    payload: RunRequest,
    db: Session = Depends(get_session),
) -> RunResponse:
    agent = db.get(RegisteredAgent, agent_id)
    if not agent:
        logger.warning("Run requested for unknown agent '%s'", agent_id)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    # Detach agent from session before the long-running invoke
    # so the DB transaction closes and does not block DDL operations
    db.expunge(agent)
    db.close()

    logger.info("Run request — agent: '%s' session: '%s'", agent_id, session_id)
    try:
        return invoke_agent(agent, session_id, payload.message, payload.json_input)
    except RuntimeError as exc:
        if _is_rate_limit_error(exc):
            logger.warning(
                "Upstream rate limit — agent: '%s' session: '%s' error: %s",
                agent_id,
                session_id,
                exc,
            )
            raise HTTPException(
                status_code=429,
                detail="Upstream LLM rate limit exceeded. Please retry later.",
            ) from exc

        logger.error("Runtime error — agent: '%s' session: '%s' error: %s", agent_id, session_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error — agent: '%s' session: '%s'", agent_id, session_id)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc