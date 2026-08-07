# ./app/models/tool.py
from typing import Any, Literal

from pydantic import BaseModel, Field


class HttpParamSchema(BaseModel):
    """Describes one field the LLM must populate when calling an HTTP tool.

    Write the description like an n8n node field description:
    precise, with expected type and an example value inline.
    """
    name: str
    type: Literal["string", "integer", "float", "boolean", "object", "array"]
    description: str
    required: bool = True
    default: Any = None


class HttpAuthConfig(BaseModel):
    type: Literal["none", "bearer", "api_key", "basic"] = "none"
    # bearer / api_key
    token: str | None = None
    token_file: str | None = None
    header_name: str = "X-API-Key"          # used when type="api_key"
    # basic
    username: str | None = None
    password: str | None = None
    password_file: str | None = None


class HttpToolConfig(BaseModel):
    """Full definition of an HTTP-based tool.

    Mirrors the n8n HTTP Request node model:
      url + method   → endpoint
      params         → what the LLM fills in to build the request body/query
      auth           → how to authenticate
      description    → injected into the tool docstring so the LLM knows
                       when and how to invoke this tool
    """
    url: str
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "POST"
    description: str
    params: list[HttpParamSchema] = Field(default_factory=list)
    auth: HttpAuthConfig = Field(default_factory=HttpAuthConfig)
    timeout_seconds: int = 30
    headers: dict[str, str] = Field(default_factory=dict)


class ToolConfig(BaseModel):
    """Generic tool entry inside an agent template.

    type="http"  → config must be a valid HttpToolConfig dict
    future types → add to TOOL_MAP in services/tool_registry.py
    """
    name: str
    type: str
    description: str | None = None
    credential_ref: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)