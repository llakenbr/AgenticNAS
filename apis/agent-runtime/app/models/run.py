from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    message: str
    json_input: dict[str, Any] | list[dict[str, Any]] | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    stream: bool = False


class RunResponse(BaseModel):
    agent_id: str
    session_id: str
    status: str
    output_text: dict[str, Any] | str
    events: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, int] | None = None
    timestamp: datetime = Field(default_factory=datetime.now)