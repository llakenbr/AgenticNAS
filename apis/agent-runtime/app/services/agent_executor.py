# app/services/agent_executor.py
"""
Stateless execution layer — the only public surface is invoke_agent().

Graph compilation is cached per agent_id + config fingerprint so the
system prompt and graph structure are compiled once on first call.
Subsequent calls for the same session_id only send a new HumanMessage;
prior conversation state is retrieved automatically from Postgres via
the PostgresSaver checkpointer keyed on thread_id = agent_id + session_id.
"""
import hashlib
import json
import uuid
import logging
from typing import Any

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage

from db.models import RegisteredAgent
from models.agent import AgentTemplateCreate
from models.run import RunResponse
from services.azure_client import get_azure_llm
from services.graph_factory import _build_system_prompt, build_graph, extract_structured_output, init_checkpointer

logger = logging.getLogger(__name__)

_graph_cache: dict[str, Any] = {}


def _build_thread_id(agent_id: str, session_id: str) -> str:
    """Scope history by agent + session to prevent cross-agent memory access."""
    return f"agent::{agent_id}::session::{session_id}"


def _estimate_prompt_tokens(llm: Any, system_prompt: str, messages: list[Any], current_message: Any) -> int | None:
    """Estimate prompt tokens for the current turn, including system prompt and history."""
    try:
        prompt_messages = [SystemMessage(content=system_prompt), *messages, current_message]
        return llm.get_num_tokens_from_messages(prompt_messages)
    except Exception as exc:
        logger.warning("Failed to estimate prompt tokens for history trim: %s", exc)
        return None


def _select_trimmed_history(
    llm: Any,
    system_prompt: str,
    messages: list[Any],
    current_message: Any,
    max_history_messages: int | None,
    max_context_tokens: int | None,
) -> list[Any]:
    """Return the suffix of messages that fits all configured history limits."""
    kept = list(messages)

    if max_history_messages is not None and len(kept) > max_history_messages:
        kept = kept[-max_history_messages:]

    if max_context_tokens is not None:
        while kept:
            prompt_tokens = _estimate_prompt_tokens(llm, system_prompt, kept, current_message)
            if prompt_tokens is None or prompt_tokens <= max_context_tokens:
                break
            kept = kept[1:]

    while kept and kept[0].__class__.__name__ == "ToolMessage":
        kept = kept[1:]

    return kept


def _collect_dangling_tool_call_removals(messages: list[Any]) -> list[RemoveMessage]:
    """Remove a trailing assistant message that still has unresolved tool calls."""
    if not messages:
        return []

    last_message = messages[-1]
    if last_message.__class__.__name__ != "AIMessage":
        return []

    tool_calls = getattr(last_message, "tool_calls", None) or []
    if not tool_calls:
        return []

    message_id = getattr(last_message, "id", None)
    if not message_id:
        logger.warning("Trailing AIMessage with tool_calls has no id; cannot prune invalid history")
        return []

    logger.warning(
        "Pruning trailing AIMessage with unresolved tool_calls from checkpoint history: %s",
        message_id,
    )
    return [RemoveMessage(id=message_id)]


def _trim_history_if_configured(
    graph: Any,
    config: dict[str, Any],
    llm: Any,
    system_prompt: str,
    current_message: Any,
    max_history_messages: int | None,
    max_context_tokens: int | None,
) -> None:
    """Repair and trim checkpointed message history before invoke."""

    if not hasattr(graph, "get_state") or not hasattr(graph, "update_state"):
        logger.warning(
            "History repair requested (max_history_messages=%s, max_context_tokens=%s), but graph state API is unavailable",
            max_history_messages,
            max_context_tokens,
        )
        return

    try:
        snapshot = graph.get_state(config)
        values = getattr(snapshot, "values", {}) or {}
        messages = values.get("messages")
        if not isinstance(messages, list):
            return

        removals = _collect_dangling_tool_call_removals(messages)

        kept_messages = _select_trimmed_history(
            llm,
            system_prompt,
            messages,
            current_message,
            max_history_messages,
            max_context_tokens,
        )

        if kept_messages != messages:
            remove_count = len(messages) - len(kept_messages)
            to_remove = messages[:remove_count]
            prefix_removals = [RemoveMessage(id=m.id) for m in to_remove if getattr(m, "id", None)]
            if remove_count and len(prefix_removals) != remove_count:
                logger.warning("History trim skipped for %d message(s) missing ids", remove_count - len(prefix_removals))
            removals.extend(prefix_removals)

        if not removals:
            return

        graph.update_state(config, {"messages": removals})
        logger.info(
            "Repaired/trimmed checkpoint history for thread_id=%s from %d message(s)",
            config.get("configurable", {}).get("thread_id", "unknown"),
            len(messages),
        )
    except Exception as exc:
        logger.warning("Failed to repair/trim history before invoke: %s", exc)


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _extract_usage_from_message(msg: Any) -> dict[str, int]:
    """Extract normalized token usage from a LangChain message."""
    usage = getattr(msg, "usage_metadata", None) or {}
    response_meta = getattr(msg, "response_metadata", {}) or {}
    provider_usage = (
        response_meta.get("token_usage")
        or response_meta.get("usage")
        or {}
    )

    input_tokens = _to_int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or provider_usage.get("input_tokens")
        or provider_usage.get("prompt_tokens")
    )
    output_tokens = _to_int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or provider_usage.get("output_tokens")
        or provider_usage.get("completion_tokens")
    )
    total_tokens = _to_int(
        usage.get("total_tokens")
        or provider_usage.get("total_tokens")
    )
    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _config_fingerprint(agent: RegisteredAgent) -> str:
    raw = json.dumps(agent.agent_config, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _collect_required_field_issues(data: Any, schema: dict[str, Any] | None, path: str = "$") -> list[str]:
    """Recursively validate required fields for the supported JSON schema subset."""
    if not schema or not isinstance(schema, dict):
        return []

    schema_type = schema.get("type")
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []

    issues: list[str] = []

    is_object_schema = schema_type == "object" or bool(properties)
    if is_object_schema:
        if not isinstance(data, dict):
            issues.append(f"{path}: expected object")
            return issues

        for key in required:
            if key not in data:
                issues.append(f"{path}.{key}: missing required field")

        for key, child_schema in properties.items():
            if key in data:
                issues.extend(_collect_required_field_issues(data[key], child_schema, f"{path}.{key}"))

        return issues

    if schema_type == "array":
        if not isinstance(data, list):
            issues.append(f"{path}: expected array")
            return issues

        item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else None
        if item_schema:
            for idx, item in enumerate(data):
                issues.extend(_collect_required_field_issues(item, item_schema, f"{path}[{idx}]"))

    return issues


def get_or_build_graph(agent: RegisteredAgent) -> Any:
    cache_key = f"{agent.agent_id}:{_config_fingerprint(agent)}"
    if cache_key not in _graph_cache:
        logger.info("Building graph for agent '%s' (cache key: %s)", agent.agent_id, cache_key)
        init_checkpointer()
        template = AgentTemplateCreate.model_validate(agent.agent_config)
        _graph_cache[cache_key] = build_graph(template)
        logger.info("Graph cached for agent '%s'", agent.agent_id)
    else:
        logger.debug("Graph cache hit for agent '%s'", agent.agent_id)
    return _graph_cache[cache_key]


def invoke_agent(
    agent: RegisteredAgent,
    session_id: str,
    message: str,
    json_input: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> RunResponse:
    logger.info(
        "Run started — agent: '%s' session: '%s' message_len: %d",
        agent.agent_id, session_id, len(message),
    )

    template = AgentTemplateCreate.model_validate(agent.agent_config)
    graph = get_or_build_graph(agent)
    llm = get_azure_llm(template.model)
    system_prompt = _build_system_prompt(template)
    thread_id = _build_thread_id(agent.agent_id, session_id)
    config = {"configurable": {"thread_id": thread_id}}
    chat_message = (
        message
        if json_input is None
        else json.dumps({"message": message, "input": json_input})
    )
    current_message = HumanMessage(content=chat_message, id=str(uuid.uuid4()))
    _trim_history_if_configured(
        graph,
        config,
        llm,
        system_prompt,
        current_message,
        template.memory.max_history_messages,
        template.memory.max_context_tokens,
    )

    try:
        result = graph.invoke(
            {"messages": [current_message]},
            config=config,
        )
    except Exception as exc:
        logger.exception(
            "Graph invoke failed — agent=%s session=%s error=%s",
            agent.agent_id,
            session_id,
            exc,
        )
        raise

    # Extract raw AIMessage text from graph result
    raw_text = ""
    tool_calls_made = []

    token_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    for msg in result.get("messages", []):
        cls = msg.__class__.__name__
        if cls == "AIMessage":
            raw_text = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)

            usage = getattr(msg, "usage_metadata", None) or {}
            if usage:
                token_usage["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
                token_usage["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
                token_usage["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
            else:
                response_meta = getattr(msg, "response_metadata", {}) or {}
                provider_usage = response_meta.get("token_usage", {}) or {}
                token_usage["input_tokens"] += int(provider_usage.get("prompt_tokens", 0) or 0)
                token_usage["output_tokens"] += int(provider_usage.get("completion_tokens", 0) or 0)
                token_usage["total_tokens"] += int(provider_usage.get("total_tokens", 0) or 0)
        elif cls == "ToolMessage":
            tool_calls_made.append(msg.name if hasattr(msg, "name") else "unknown")

    logger.info(
        "Graph completed — tool calls made: %s",
        tool_calls_made if tool_calls_made else "NONE (check prompt enforcement)",
    )

    # Phase 2 — prefer LangGraph structured_response from graph state
    structured = result.get("structured_response")

    if structured is not None:
        logger.info(
            "Structured response from graph node — type=%s",
            type(structured).__name__,
        )

        parsed_raw = None
        if raw_text:
            try:
                parsed_raw = json.loads(raw_text)
                logger.info(
                    "Structured response reparsed from raw AIMessage content — keys=%s",
                    list(parsed_raw.keys()) if isinstance(parsed_raw, dict) else "non-dict",
                )
            except json.JSONDecodeError:
                logger.warning(
                    "Could not parse raw AIMessage content as JSON; falling back to structured_response"
                )

        if isinstance(structured, dict):
            final_output = structured
        elif hasattr(structured, "model_dump"):
            final_output = structured.model_dump(exclude_none=True)
        else:
            final_output = {"raw_text": str(structured)}

        # Compatibility alias for legacy outputs.
        if isinstance(final_output, dict) and "name" in final_output and "model_name" not in final_output:
            final_output["model_name"] = final_output.pop("name")

        structured_issues = _collect_required_field_issues(final_output, template.output_schema)
        raw_issues = (
            _collect_required_field_issues(parsed_raw, template.output_schema)
            if isinstance(parsed_raw, dict)
            else ["$: raw output not a JSON object"]
        )

        # Fall back to reparsed raw AI JSON only if structured output violates schema
        # and raw output satisfies required-field constraints recursively.
        if isinstance(parsed_raw, dict) and structured_issues and not raw_issues:
            logger.warning(
                "Structured response violates required-field schema; falling back to reparsed raw AI JSON"
            )
            if "name" in parsed_raw and "model_name" not in parsed_raw:
                parsed_raw["model_name"] = parsed_raw.pop("name")
            final_output = parsed_raw

        logger.info(
            "Structured response from graph node — output_type=%s",
            type(final_output).__name__,
        )
    else:
        if template.output_schema:
            logger.warning(
                "structured_response absent from graph state — running fallback extraction"
            )

        fallback_text = extract_structured_output(raw_text, template.output_schema, template)
        try:
            final_output = json.loads(fallback_text)
        except json.JSONDecodeError:
            final_output = {"raw_text": fallback_text}

    schema_issues = _collect_required_field_issues(final_output, template.output_schema)
    if schema_issues:
        preview = "; ".join(schema_issues[:10])
        if len(schema_issues) > 10:
            preview += f"; ... (+{len(schema_issues) - 10} more)"
        raise RuntimeError(f"Output schema validation failed: {preview}")

    logger.info(
        "Run completed — agent: '%s' session: '%s' response_len: %d",
        agent.agent_id, session_id, len(final_output),
    )

    return RunResponse(
        agent_id=agent.agent_id,
        session_id=session_id,
        status="completed",
        output_text=final_output,
        events=[
            {"type": "run.started",   "agent_id": agent.agent_id, "session_id": session_id},
            {"type": "run.tool_calls", "tools": tool_calls_made},
            {"type": "run.completed", "agent_id": agent.agent_id, "session_id": session_id},
        ],
        token_usage=token_usage,
    )


def clear_cache(agent_id: str) -> None:
    """Invalidate graph cache for this agent."""
    cache_key = f"{agent_id}:*"  # or compute the full key
    invalidated = 0
    for key in list(_graph_cache.keys()):
        if key.startswith(agent_id + ":"):
            del _graph_cache[key]
            invalidated += 1
    logger.info("Graph cache cleared for agent '%s' — invalidated %d entries", agent_id, invalidated)
