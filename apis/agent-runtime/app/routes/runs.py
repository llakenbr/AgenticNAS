# ./app/routes/runs.py
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.models import RegisteredAgent
from db.session import get_session
from models.run import RunRequest, RunResponse
from services.agent_executor import invoke_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents/{agent_id}/sessions/{session_id}", tags=["runs"])


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
        return invoke_agent(agent, session_id, payload.message)
    except RuntimeError as exc:
        logger.error("Runtime error — agent: '%s' session: '%s' error: %s", agent_id, session_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error — agent: '%s' session: '%s'", agent_id, session_id)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc