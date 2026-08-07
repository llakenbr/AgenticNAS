import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from db.models import SessionEpisode, SessionState
from db.session import get_session
from models.memory import EpisodeResponse, EpisodeWrite, StateResponse, StateWrite

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["memory"])


def _get_latest_state(db: Session, session_id: str) -> SessionState | None:
    """Return the newest state snapshot for a session."""
    stmt = (
        select(SessionState)
        .where(SessionState.session_id == session_id)
        .order_by(desc(SessionState.created_at), desc(SessionState.state_id))
        .limit(1)
    )
    return db.scalar(stmt)


def _seed_session_if_missing(db: Session, session_id: str) -> None:
    """Ensure a session has at least one state row for compatibility flows."""
    if _get_latest_state(db, session_id) is not None:
        return

    state = SessionState(session_id=session_id, summary_state="")
    db.add(state)
    db.flush()


@router.post("/episodes", response_model=EpisodeResponse)
def record_episode(
    payload: EpisodeWrite,
    db: Session = Depends(get_session),
) -> EpisodeResponse:
    """Record an episodic event from agent execution.

    POST /episodes with body:
    {
        "session_id": "...",
        "agent": "model-builder",
        "message": "...",
        "episode_metadata": {...}
    }
    """
    # Ensure session exists
    _seed_session_if_missing(db, payload.session_id)

    # Create episode
    episode = SessionEpisode(
        session_id=payload.session_id,
        agent=payload.agent,
        message=payload.message,
        episode_metadata=payload.episode_metadata or {},
    )
    db.add(episode)
    db.commit()
    db.refresh(episode)

    logger.info(
        f"Recorded episode {episode.episode_id} for session {payload.session_id} step {episode.step_index} agent {payload.agent}"
    )

    return EpisodeResponse(
        episode_id=episode.episode_id,
        session_id=episode.session_id,
        step_index=episode.step_index,
        agent=episode.agent,
        message=episode.message,
        episode_metadata=episode.episode_metadata or {},
        created_at=episode.created_at,
    )


@router.post("/state", response_model=StateResponse)
def write_state(
    payload: StateWrite,
    db: Session = Depends(get_session),
) -> StateResponse:
    """Append a per-session state snapshot.

    POST /state with body:
    {
        "session_id": "...",
        "summary_text": "Session summary...",
        "summary_state": "exploration"
    }
    """
    state = SessionState(
        session_id=payload.session_id,
        summary_text=payload.summary_text,
        summary_state=payload.summary_state or "",
    )
    db.add(state)

    db.commit()
    db.refresh(state)

    logger.info(f"Appended state snapshot for session {payload.session_id}")

    return StateResponse(
        session_id=state.session_id,
        summary_text=state.summary_text,
        summary_state=state.summary_state or "",
        created_at=state.created_at,
        updated_at=state.updated_at,
    )


@router.get("/state", response_model=StateResponse)
def get_state(
    session_id: str = Query(..., description="Session ID"),
    db: Session = Depends(get_session),
) -> StateResponse:
    """Retrieve latest state snapshot for a session.

    GET /state?session_id=...
    """
    state = _get_latest_state(db, session_id)
    if state is None:
        return StateResponse(
            session_id=session_id,
            summary_text=None,
            summary_state="",
            created_at=None,
            updated_at=None,
        )

    return StateResponse(
        session_id=state.session_id,
        summary_text=state.summary_text,
        summary_state=state.summary_state or "",
        created_at=state.created_at,
        updated_at=state.updated_at,
    )

