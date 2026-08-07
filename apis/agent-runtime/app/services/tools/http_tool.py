# app/services/tools/http_tool.py
"""
Builds a LangChain StructuredTool from a ToolConfig with type='http'.

Auth resolution order (mirrors settings._read_secret):
  token_file  > token    for bearer / api_key
  password_file > password  for basic
"""
from __future__ import annotations

import logging

import base64
import json
from pathlib import Path
from typing import Any, Type

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from models.tool import HttpAuthConfig, HttpToolConfig, HttpParamSchema, ToolConfig

logger = logging.getLogger(__name__)

# ── Secret resolution ─────────────────────────────────────────────────────────

def _resolve_secret(value: str | None, file_path: str | None) -> str | None:
    if file_path:
        p = Path(file_path)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8").strip()
    return value


# ── Dynamic Pydantic input model ──────────────────────────────────────────────

_PYTHON_TYPES: dict[str, Any] = {
    "string":  str,
    "integer": int,
    "float":   float,
    "boolean": bool,
    "object":  dict,
    "array":   list,
}


def _build_input_model(tool_name: str, params: list[HttpParamSchema]) -> Type[BaseModel]:
    """Dynamically create a Pydantic model the LLM uses to populate the request."""
    model_fields: dict[str, Any] = {}

    for p in params:
        base_type = _PYTHON_TYPES.get(p.type, Any)

        # OpenAI's parse-based tool validation is strict and can reject free-form
        # object properties. Expose object arguments as JSON strings at the schema
        # layer, then decode back to Python objects inside the HTTP caller.
        if p.type == "object":
            base_type = str

        # OpenAI's function schema validator requires deterministic required/property
        # alignment. Keep every declared parameter in `required`; optional inputs are
        # represented as nullable types instead.
        if p.required:
            python_type = base_type
        else:
            python_type = base_type | None  # type: ignore[operator]

        model_fields[p.name] = (
            python_type,
            Field(..., description=p.description),
        )

    # If no explicit params: single optional 'body', but ALLOW extra fields
    if not model_fields:
        model_fields["body"] = (
            dict | None,
            Field(
                ...,
                description="Optional full JSON body. If omitted, all other fields are treated as the body.",
            ),
        )

    model = create_model(f"{tool_name}_Input", **model_fields)
    # Explicit schemas should be strict for OpenAI tool parsing compatibility.
    model.model_config["extra"] = "forbid"
    _validate_openai_schema_compat(model, tool_name)
    return model


def _validate_openai_schema_compat(model: Type[BaseModel], tool_name: str) -> None:
    """Fail fast on schema shapes OpenAI rejects for function tools."""
    schema = model.model_json_schema()
    properties = schema.get("properties") or {}
    required = schema.get("required") or []

    if not isinstance(properties, dict):
        raise RuntimeError(
            f"Invalid tool schema for '{tool_name}': properties must be an object"
        )

    if not isinstance(required, list):
        raise RuntimeError(
            f"Invalid tool schema for '{tool_name}': required must be an array"
        )

    extra_required = [name for name in required if name not in properties]
    if extra_required:
        raise RuntimeError(
            f"Invalid tool schema for '{tool_name}': required contains non-property keys {extra_required}"
        )


# ── Auth headers ──────────────────────────────────────────────────────────────

def _build_auth_headers(auth: HttpAuthConfig) -> dict[str, str]:
    if auth.type == "none":
        return {}
    if auth.type == "bearer":
        token = _resolve_secret(auth.token, auth.token_file)
        if not token:
            raise RuntimeError("Bearer token not configured.")
        return {"Authorization": f"Bearer {token}"}
    if auth.type == "api_key":
        token = _resolve_secret(auth.token, auth.token_file)
        if not token:
            raise RuntimeError("API key not configured.")
        return {auth.header_name: token}
    if auth.type == "basic":
        password = _resolve_secret(auth.password, auth.password_file) or ""
        creds = base64.b64encode(f"{auth.username}:{password}".encode()).decode()
        return {"Authorization": f"Basic {creds}"}
    return {}


# ── HTTP caller ───────────────────────────────────────────────────────────────

def _make_http_caller(cfg: HttpToolConfig, tool_name: str):  # ← add tool_name
    def _call(**kwargs: Any) -> str:
        if "body" in kwargs:
            raw_body = kwargs["body"]
            if isinstance(raw_body, str):
                try:
                    body = json.loads(raw_body)
                except json.JSONDecodeError:
                    body = {"body": raw_body}
            elif isinstance(raw_body, (dict, list)):
                body = raw_body
            else:
                body = raw_body
        else:
            body = {k: v for k, v in kwargs.items() if v is not None}

        logger.info(
            "HTTP_TOOL_CALL start — name=%s method=%s url=%s args=%s",
            tool_name,   # ← use tool_name, not cfg.name
            cfg.method,
            cfg.url,
            list(body.keys()),
        )
        logger.debug("HTTP_TOOL_CALL body=%s", body)

        headers = {"Content-Type": "application/json", **cfg.headers}
        try:
            headers.update(_build_auth_headers(cfg.auth))
        except RuntimeError as exc:
            logger.warning("HTTP_TOOL_CALL auth error — tool=%s error=%s", tool_name, exc)
            return f"[auth error] {exc}"

        try:
            if cfg.method == "GET":
                resp = requests.get(
                    cfg.url, params=body, headers=headers, timeout=cfg.timeout_seconds
                )
            else:
                resp = requests.request(
                    cfg.method, cfg.url, json=body,
                    headers=headers, timeout=cfg.timeout_seconds,
                )
        except requests.exceptions.RequestException as exc:
            logger.error("HTTP_TOOL_CALL http error — tool=%s error=%s", tool_name, exc)
            raise RuntimeError(
                f"HTTP transport error from {tool_name}: {exc}"
            )

        logger.info(
            "HTTP_TOOL_CALL done — name=%s status=%s",
            tool_name,   # ← same here
            resp.status_code,
        )

        if not resp.ok:
            try:
                payload = resp.json()
            except Exception:
                payload = resp.text

            if isinstance(payload, dict) and "detail" in payload:
                detail = payload["detail"]
            else:
                detail = payload

            logger.warning(
                "HTTP_TOOL_CALL non-2xx — name=%s status=%s detail=%s",
                tool_name,
                resp.status_code,
                detail,
            )
            retryable = resp.status_code in {400, 422}
            if not retryable:
                raise RuntimeError(
                    f"HTTP {resp.status_code} from {tool_name}: {detail}"
                )
            return json.dumps(
                {
                    "ok": False,
                    "retryable": retryable,
                    "status": resp.status_code,
                    "tool": tool_name,
                    "detail": detail,
                },
                ensure_ascii=False,
            )

        try:
            return json.dumps(resp.json(), indent=2)
        except Exception:
            return resp.text

    return _call


# ── Public factory ────────────────────────────────────────────────────────────

def build_http_tool(tool_cfg: ToolConfig) -> StructuredTool:
    http_cfg = HttpToolConfig.model_validate(tool_cfg.config)
    input_model = _build_input_model(tool_cfg.name, http_cfg.params)
    caller = _make_http_caller(http_cfg, tool_cfg.name)  # ← pass name here
    return StructuredTool.from_function(
        func=caller,
        name=tool_cfg.name,
        description=http_cfg.description,
        args_schema=input_model,
        return_direct=False,
        handle_tool_error=True,
    )