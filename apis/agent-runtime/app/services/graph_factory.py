# app/services/graph_factory.py
"""
Build a LangChain agent graph from an AgentTemplateCreate config.

Two-phase output:
    Phase 1 — Agent graph: LLM + tools, runs until termination.
    Phase 2 — Structured extraction: separate LLM call enforces output_schema.
"""
import json
import logging
import re
from typing import Any, Annotated, Literal, TypedDict
from pydantic import BaseModel, Field, create_model

import psycopg
from langchain_core.messages import AIMessage
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver

from core.settings import settings
from models.agent import AgentTemplateCreate
from services.azure_client import get_azure_llm
from services.tool_registry import resolve_tools

logger = logging.getLogger(__name__)

_POSTGRES_CONNECT_TIMEOUT = 10
_checkpointer: PostgresSaver | None = None


def init_checkpointer() -> None:
    global _checkpointer
    if _checkpointer is not None:
        return
    logger.info("Initialising checkpointer — connecting to Postgres")
    try:
        conn = psycopg.connect(
            settings.postgres_graph_dsn,
            autocommit=True,
            connect_timeout=_POSTGRES_CONNECT_TIMEOUT,
        )
    except psycopg.OperationalError as exc:
        logger.error("Checkpointer Postgres connection failed: %s", exc)
        raise RuntimeError(f"Checkpointer Postgres connection failed: {exc}") from exc
    saver = PostgresSaver(conn)
    logger.info("Running checkpointer DDL setup")
    saver.setup()
    _checkpointer = saver
    logger.info("Checkpointer ready")


def _get_checkpointer() -> PostgresSaver:
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialised — call init_checkpointer() at startup")
    return _checkpointer


class _NoToolGraphState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    structured_response: Any


def _build_no_tool_graph(
    template: AgentTemplateCreate,
    llm: Any,
    system_prompt: str,
    response_format_schema: dict[str, Any] | None,
) -> Any:
    """Build a one-node graph for agents that do not use tools.

    This avoids LangChain's tool-choice request path entirely, which is required
    for Azure deployments that reject `tool_choice` when no tools are present.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("messages"),
    ])

    model = (
        llm.with_structured_output(response_format_schema, include_raw=True)
        if response_format_schema is not None
        else llm
    )
    chain = prompt | model

    def _call_model(state: _NoToolGraphState) -> dict[str, Any]:
        result = chain.invoke({"messages": state.get("messages", [])})

        if response_format_schema is None:
            return {"messages": [result]}

        raw_message = result.get("raw") if isinstance(result, dict) else None
        parsed_result = result.get("parsed") if isinstance(result, dict) else result

        if parsed_result is None and raw_message is not None:
            raw_content = getattr(raw_message, "content", None)
            if isinstance(raw_content, str):
                parsed_result = _extract_json_from_text(raw_content) or {"raw_text": raw_content}
            else:
                parsed_result = raw_content

        normalized = _normalize_structured_result(parsed_result)
        message_kwargs: dict[str, Any] = {}
        if raw_message is not None:
            usage_metadata = getattr(raw_message, "usage_metadata", None)
            response_metadata = getattr(raw_message, "response_metadata", None) or {}
            if usage_metadata is not None:
                message_kwargs["usage_metadata"] = usage_metadata
            if response_metadata:
                message_kwargs["response_metadata"] = response_metadata

        return {
            "messages": [AIMessage(content=json.dumps(normalized, ensure_ascii=False), **message_kwargs)],
            "structured_response": normalized,
        }

    graph_builder = StateGraph(_NoToolGraphState)
    graph_builder.add_node("model", _call_model)
    graph_builder.add_edge(START, "model")
    graph_builder.add_edge("model", END)
    return graph_builder.compile(checkpointer=_get_checkpointer())


def _preflight_openai_tool_schemas(tools: list[BaseTool]) -> None:
    """Validate OpenAI-converted tool schemas before the first model call."""
    for tool in tools:
        try:
            payload = convert_to_openai_tool(tool)
            params = payload.get("function", {}).get("parameters", {})
            properties = params.get("properties") or {}
            required = params.get("required") or []
            if isinstance(required, list) and isinstance(properties, dict):
                extra_required = [name for name in required if name not in properties]
                if extra_required:
                    raise ValueError(
                        f"required contains keys not present in properties: {extra_required}"
                    )
        except Exception as exc:
            raise RuntimeError(
                f"Tool schema preflight failed for '{tool.name}': {exc}"
            ) from exc


def build_graph(template: AgentTemplateCreate) -> Any:
    logger.info("build_graph: resolving LLM + tools")
    llm = get_azure_llm(template.model)
    tools: list[BaseTool] = resolve_tools(template.tools)
    _preflight_openai_tool_schemas(tools)

    logger.info(
        "build_graph: prompt.system length=%d, system_context=%d, tools=%s",
        len(template.prompt.system),
        len(template.prompt.system_context),
        [t.name for t in tools],
    )
    logger.debug("build_graph: prompt.system=%r", template.prompt.system)
    system_prompt = _build_system_prompt(template)

    if not tools:
        logger.info("build_graph: no tools resolved — using direct LLM graph path")
        response_format_schema = getattr(template, "output_schema", None)
        if response_format_schema is not None:
            logger.info("build_graph: no-tool path using raw JSON schema response_format")
        else:
            logger.info("build_graph: no output_schema — structured node disabled")

        graph = _build_no_tool_graph(template, llm, system_prompt, response_format_schema)
        logger.info("build_graph: done")
        return graph

    response_format_model = build_response_format_model(
        getattr(template, "output_schema", None)
    )

    if response_format_model is not None:
        logger.info(
            "build_graph: response_format=%s (main model handles extraction)",
            response_format_model.__name__,
        )
    else:
        logger.info("build_graph: no output_schema — structured node disabled")

    logger.info("build_graph: compiling agent graph")
    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=_get_checkpointer(),
        response_format=response_format_model,  # ← Pydantic class only, never a tuple
        name=template.agent_id,
    )
    logger.info("build_graph: done")
    return graph


def extract_structured_output(
    raw_text: str,
    output_schema: dict | None,
    template: AgentTemplateCreate,
) -> str:
    """Fallback Phase 2 — only called when LangGraph structured node missed.

    Normal flow: LangGraph generate_structured_response node handles this.
    This runs only if result['structured_response'] is absent from graph state.
    """
    if not output_schema:
        return raw_text

    parsed = _extract_json_from_text(raw_text)
    if parsed is not None and _matches_schema(parsed, output_schema):
        logger.info("Fallback: output already matches schema — skipping extraction call")
        return json.dumps(parsed)

    logger.warning(
        "Fallback structured extraction running — graph node missed or output_schema_model not set"
    )
    llm = get_azure_llm(template.output_schema_model or template.model)

    structured_llm = llm.with_structured_output(output_schema)

    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a JSON extraction assistant. "
            "Extract and return structured data from the provided text "
            "that matches the required schema exactly. "
            "Do not add commentary."
        )),
        ("human", "Extract structured output from:\n\n{text}"),
    ])

    try:
        result = (extraction_prompt | structured_llm).invoke({"text": raw_text})

        normalized = _normalize_structured_result(result)
        out = json.dumps(normalized)

        logger.info("Schema extraction completed — output_len: %d", len(out))
        return out
    except Exception as exc:
        logger.warning("Structured output extraction failed (%s) — returning raw text", exc)
        return raw_text


def build_response_format_model(output_schema: dict | None) -> type[BaseModel] | None:
    """Convert output_schema config dict → Pydantic model for LangGraph response_format.

    Returns None if no schema defined — disables the structured output node entirely.
    """
    if not output_schema:
        return None
    return _build_output_model_from_schema(output_schema)


def _build_output_model_from_schema(output_schema: dict) -> type[BaseModel]:
    title = output_schema.get("title", "AgentOutputSchema")
    properties = output_schema.get("properties", {}) if isinstance(output_schema.get("properties"), dict) else {}
    required = set(output_schema.get("required", [])) if isinstance(output_schema.get("required"), list) else set()

    def _annotation_from_json_schema(schema: dict, field_key: str, parent_name: str) -> Any:
        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            return Literal[tuple(enum_values)]

        json_type = schema.get("type")
        if json_type == "object" or isinstance(schema.get("properties"), dict):
            nested_props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
            if nested_props:
                nested_required = schema.get("required") if isinstance(schema.get("required"), list) else []
                nested_title = schema.get("title") or f"{parent_name}_{field_key}"
                return _build_output_model_from_schema({
                    "title": nested_title,
                    "type": "object",
                    "properties": nested_props,
                    "required": nested_required,
                    "additionalProperties": schema.get("additionalProperties", False),
                })
            return dict[str, Any]

        if json_type == "array":
            items_schema = schema.get("items") if isinstance(schema.get("items"), dict) else {}
            item_ann = _annotation_from_json_schema(items_schema, f"{field_key}_item", parent_name)
            return list[item_ann]

        primitive_map: dict[str, Any] = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
        }
        return primitive_map.get(json_type, Any)

    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            continue
        description = field_schema.get("description", "")
        annotation = _annotation_from_json_schema(field_schema, field_name, title)

        if field_name in required:
            fields[field_name] = (annotation, Field(..., description=description))
        else:
            fields[field_name] = (annotation | None, Field(default=None, description=description))

    model = create_model(title, **fields)
    model.model_config["extra"] = "forbid" if output_schema.get("additionalProperties") is False else "allow"
    return model


def _make_strict_object_model(name: str, description: str, nested_schema: dict) -> type[BaseModel]:
    """Create an additionalProperties=false sub-model for an object field.

    If the nested schema has its own `properties`, use them.
    Otherwise, fall back to a single free-text `value` field — this keeps
    Azure happy while still accepting arbitrary JSON from the LLM.
    """
    nested_props = nested_schema.get("properties", {})

    if nested_props:
        # Recurse for explicitly defined nested properties
        return _build_output_model_from_schema({
            "title": name,
            "properties": nested_props,
            "required": nested_schema.get("required", list(nested_props.keys())),
        })

    # No sub-properties defined — single JSON-string field as passthrough
    # The LLM will serialize the object to a string; you parse it downstream
    class _StrictObject(BaseModel):
        model_config = {"extra": "forbid"}
        value: str = Field(
            default="{}",
            description=f"Serialized JSON string for: {description}",
        )

    _StrictObject.__name__ = name
    _StrictObject.__qualname__ = name
    return _StrictObject


def _unwrap_string_fields(data: dict) -> dict:
    """Parse JSON-string values back to native objects (recursive).

    Handles:
    - _StrictObject: {"value": "{...}"} → dict
    - Double-escaped JSON strings from LLM: "{\\\"key\\\":..." → {"key": ...}
    """
    result = {}
    for k, v in data.items():
        if isinstance(v, dict):
            if list(v.keys()) == ["value"] and isinstance(v["value"], str):
                try:
                    result[k] = json.loads(v["value"])
                except json.JSONDecodeError:
                    result[k] = v
            else:
                result[k] = _unwrap_string_fields(v)
        elif isinstance(v, str):
            # Double-escaped JSON from LLM: "{\\\"key\\\":{\\\"subkey\\\":...}"
            # First try direct parse (handles normal cases)
            try:
                result[k] = json.loads(v)
            except json.JSONDecodeError:
                # Fallback: unescape the double-backslashes
                unescaped = v.encode("utf-8").decode("unicode_escape")
                try:
                    result[k] = json.loads(unescaped)
                except json.JSONDecodeError:
                    result[k] = v
        elif isinstance(v, list):
            result[k] = [
                _unwrap_string_fields(i) if isinstance(i, dict) else i
                for i in v
            ]
        else:
            result[k] = v
    return result


def _normalize_structured_result(result: Any) -> dict[str, Any]:
    """Convert structured-output results into a plain JSON-serializable dict.

    This avoids Pydantic serializer warnings when LangChain returns wrapped
    parsed objects or nested Pydantic models.
    Also unwraps any JSON-string fields introduced by _StrictObject or
    string-typed schema fields back to native Python objects before serialization.
    """
    logger.debug(
        "_normalize: raw nodes='%s' type=%s len=%d",
        str(data.get("nn_taxonomy", {}).get("nodes", ""))[:200],
        type(data.get("nn_taxonomy", {}).get("nodes")),
        len(str(data.get("nn_taxonomy", {}).get("nodes", "")))
    )
    if isinstance(result, dict):
        unwrapped = _unwrap_string_fields(result)
        logger.debug(
            "_normalize: unwrapped nodes keys=%s",
            list(unwrapped.get("nn_taxonomy", {}).get("nodes", {}).keys())
            if isinstance(unwrapped.get("nn_taxonomy", {}).get("nodes"), dict)
            else "not dict",
        )
        return unwrapped

    if hasattr(result, "model_dump"):
        data = result.model_dump(exclude_none=True)

        # LangChain may wrap the actual parsed object like:
        # {"raw": ..., "parsed": AgentOutputSchema(...), "parsing_error": None}
        if isinstance(data, dict) and "parsed" in data and data["parsed"] is not None:
            parsed = data["parsed"]
            if hasattr(parsed, "model_dump"):
                data = parsed.model_dump(exclude_none=True)
            elif isinstance(parsed, dict):
                data = parsed

        logger.debug("_normalize: unwrapped nodes keys=%s", list(data.get("nn_taxonomy", {}).get("nodes", {}).keys()) if isinstance(data.get("nn_taxonomy", {}).get("nodes"), dict) else "not dict")
        return _unwrap_string_fields(data)

    if hasattr(result, "__dict__"):
        data = dict(result.__dict__)
        if "parsed" in data and data["parsed"] is not None:
            parsed = data["parsed"]
            if hasattr(parsed, "model_dump"):
                data = parsed.model_dump(exclude_none=True)
            elif isinstance(parsed, dict):
                data = parsed
        logger.debug("_normalize: unwrapped nodes keys=%s", list(data.get("nn_taxonomy", {}).get("nodes", {}).keys()) if isinstance(data.get("nn_taxonomy", {}).get("nodes"), dict) else "not dict")
        return _unwrap_string_fields(data)

    raise TypeError(f"Unsupported structured result type: {type(result)}")


def _extract_json_from_text(text: str) -> dict | None:
    """Parse a JSON object from raw text, stripping markdown fences."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    return None


def _matches_schema(data: dict, schema: dict) -> bool:
    """Lightweight schema check — required fields present, no extra fields if strict.

    Does not recurse into nested objects. Sufficient to gate the fast-path.
    """
    required = schema.get("required", [])
    if not all(field in data for field in required):
        logger.debug(
            "Schema fast-path miss — missing required fields: %s",
            [f for f in required if f not in data],
        )
        return False

    if schema.get("additionalProperties") is False:
        allowed = set(schema.get("properties", {}).keys())
        extra = set(data.keys()) - allowed
        if extra:
            logger.debug("Schema fast-path miss — extra fields not in schema: %s", extra)
            return False

    return True


def _escape_prompt_braces(text: str) -> str:
    """Escape single braces for ChatPromptTemplate, idempotent."""
    text = text.replace("{{", "\x00OPEN\x00").replace("}}", "\x00CLOSE\x00")
    text = text.replace("{", "{{").replace("}", "}}")
    text = text.replace("\x00OPEN\x00", "{{").replace("\x00CLOSE\x00", "}}")
    return text


def _build_system_prompt(template: AgentTemplateCreate) -> str:
    system_parts = [template.prompt.system]
    for ctx in template.prompt.system_context:
        system_parts.append(ctx)
    if template.prompt.developer:
        system_parts.append(template.prompt.developer)

    combined = "\n\n".join(system_parts)
    escaped = _escape_prompt_braces(combined)

    logger.debug("System prompt: %d block(s), %d chars (escaped)", len(system_parts), len(escaped))
    return escaped