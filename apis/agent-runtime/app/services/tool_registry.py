# .app/services/tool_registry.py
"""
Maps ToolConfig.type strings to factory functions.

To add a new tool type:
  1. Add a builder in services/tools/
  2. Import it here and register in TOOL_MAP
"""
from typing import Callable

from langchain_core.tools import BaseTool

from models.tool import ToolConfig
from services.tools.http_tool import build_http_tool


TOOL_MAP: dict[str, Callable[[ToolConfig], BaseTool]] = {
    "http": build_http_tool,
}


def resolve_tools(tool_configs: list[ToolConfig]) -> list[BaseTool]:
    resolved: list[BaseTool] = []
    for cfg in tool_configs:
        factory = TOOL_MAP.get(cfg.type)
        if factory is None:
            raise ValueError(
                f"Unknown tool type '{cfg.type}' for tool '{cfg.name}'. "
                f"Registered types: {list(TOOL_MAP.keys())}"
            )
        resolved.append(factory(cfg))
    return resolved