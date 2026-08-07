# ./app/models/agent.py
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from models.tool import ToolConfig


class ModelConfig(BaseModel):
    provider: Literal["azure_openai", "custom"]
    model_name: str
    temperature: float = 0.0
    streaming: bool = False
    credential_ref: str | None = None


class PromptConfig(BaseModel):
    system: str
    system_context: list[str] = Field(
        default_factory=list,
        description=(
            "Additional system-role messages injected after 'system', in order. "
            "Each entry is a separate SystemMessage — use for large fixed context "
            "(specs, templates) to avoid JSON escaping issues in the registration payload."
        ),
    )
    developer: str | None = None


class MemoryConfig(BaseModel):
    backend: Literal["postgres"] = "postgres"
    thread_table: str | None = None
    max_history_messages: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional cap for persisted chat messages loaded per thread. "
            "When set, runtime trims older checkpoint messages and keeps only the latest N."
        ),
    )
    max_context_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional cap for estimated prompt tokens (system prompt + persisted chat history + current input). "
            "When set, runtime removes oldest checkpoint messages until the estimated prompt fits."
        ),
    )


class AgentTemplateCreate(BaseModel):
    agent_id: str
    template_id: str | None = None
    name: str
    description: str | None = None
    model: ModelConfig
    tools: list[ToolConfig] = Field(default_factory=list)
    prompt: PromptConfig
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON Schema dict. If set, graph_factory binds structured output to the final LLM call."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTemplateResponse(BaseModel):
    agent_id: str
    template_id: str | None = None
    status: str
    loaded: bool
    created_at: datetime = Field(default_factory=datetime.now())
    metadata: dict[str, Any] = Field(default_factory=dict)