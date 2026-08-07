from datetime import datetime
import json
import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator


logger = logging.getLogger(__name__)


class EpisodeWrite(BaseModel):
    """Record an episodic event from agent execution."""
    session_id: str
    agent: str
    message: str
    episode_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message", mode="before")
    @classmethod
    def _normalize_message(cls, value: Any) -> str:
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True)
        if isinstance(value, str):
            return value
        raise TypeError("message must be a string or an object")


class StateWrite(BaseModel):
    """Append a per-session state snapshot."""
    session_id: str
    summary_text: str | dict[str, Any] | None = None
    summary_state: str = ""

    @field_validator("summary_text", mode="before")
    @classmethod
    def _normalize_summary_text(cls, value: Any) -> str | None:
        if value is None:
            logger.debug("StateWrite.summary_text received None")
            return None
        if isinstance(value, str):
            logger.debug("StateWrite.summary_text accepted string input")
            return value
        if isinstance(value, dict):
            logger.info("StateWrite.summary_text coerced dict to JSON string")
            return json.dumps(value, sort_keys=True)
        logger.info(
            "StateWrite.summary_text coerced %s to string via str()",
            type(value).__name__,
        )
        return str(value)

    @field_validator("summary_state", mode="before")
    @classmethod
    def _normalize_summary_state(cls, value: Any) -> str:
        if value is None or value == "":
            logger.debug("StateWrite.summary_state received empty value; normalized to empty string")
            return ""
        if isinstance(value, str):
            logger.debug("StateWrite.summary_state accepted string input")
            return value
        if isinstance(value, dict):
            logger.info("StateWrite.summary_state coerced dict to JSON string")
            return json.dumps(value, sort_keys=True)
        if isinstance(value, (list, tuple, set)):
            logger.info(
                "StateWrite.summary_state coerced %s to JSON string",
                type(value).__name__,
            )
            return json.dumps(list(value), sort_keys=True)
        if isinstance(value, (int, float, bool)):
            logger.info(
                "StateWrite.summary_state coerced scalar %s to string",
                type(value).__name__,
            )
            return str(value)
        logger.warning(
            "StateWrite.summary_state received unsupported type %s; coerced via str()",
            type(value).__name__,
        )
        return str(value)


class StateResponse(BaseModel):
    """Retrieve latest summary + structured state for orchestrator prompts."""
    session_id: str
    summary_text: str | None
    summary_state: str
    created_at: datetime | None
    updated_at: datetime | None


class EpisodeResponse(BaseModel):
    """Episode record from storage."""
    episode_id: int
    session_id: str
    step_index: int
    agent: str
    message: str
    episode_metadata: dict[str, Any]
    created_at: datetime
