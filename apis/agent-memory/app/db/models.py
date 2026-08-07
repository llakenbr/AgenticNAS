from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from db.session import Base


class SessionState(Base):
    """Per-session state snapshot: summary text + structured JSON state."""

    __tablename__ = "agent_session_states"

    state_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_state: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class SessionEpisode(Base):
    """Episodic event from agent execution: step_index, agent, message, metadata."""

    __tablename__ = "agent_session_episodes"
    __table_args__ = (
        UniqueConstraint("session_id", "step_index", name="uq_agent_session_episodes_session_step"),
    )

    episode_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    agent: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    episode_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
