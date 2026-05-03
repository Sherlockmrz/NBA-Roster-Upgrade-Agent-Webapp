"""Safe OpenRouter client helpers with deterministic fallback behavior."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv
import requests


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openrouter/free"
DEFAULT_TIMEOUT_SECONDS = 30
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"
MISSING_KEY_WARNING = "OpenRouter API key is missing; deterministic fallback was used."


@dataclass
class LLMStatus:
    """Runtime LLM availability and warning state."""

    available: bool = False
    model: str = DEFAULT_OPENROUTER_MODEL
    key_preview: str = ""
    site_url: str = ""
    app_name: str = ""
    dotenv_path: str = str(DOTENV_PATH)
    warnings: list[str] = field(default_factory=list)


@dataclass
class LLMCallResult:
    """Structured result for one OpenRouter call."""

    ok: bool
    content: Any = None
    model: str = DEFAULT_OPENROUTER_MODEL
    used_fallback: bool = False
    error_type: str = ""
    error_message: str = ""
    http_status: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "content": self.content,
            "model": self.model,
            "used_fallback": self.used_fallback,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "http_status": self.http_status,
        }


_STATUS = LLMStatus()
_LAST_CALL_RESULT = LLMCallResult(ok=False, used_fallback=True)
_CALL_HISTORY: list[LLMCallResult] = []


def get_llm_status() -> LLMStatus:
    """Return a copy of the most recent LLM status."""

    _refresh_status()
    return deepcopy(_STATUS)


def is_llm_available() -> bool:
    """Return whether an OpenRouter API key is configured."""

    return get_llm_status().available


def clear_llm_call_history() -> None:
    """Clear structured call results for the next user-triggered run."""

    _CALL_HISTORY.clear()


def get_last_llm_call_result() -> LLMCallResult:
    """Return a copy of the most recent structured call result."""

    return deepcopy(_LAST_CALL_RESULT)


def get_llm_call_history() -> list[LLMCallResult]:
    """Return structured results for recent LLM calls."""

    return deepcopy(_CALL_HISTORY)


def call_llm_json(messages: list[dict[str, str]], fallback: Any) -> Any:
    """Call OpenRouter and parse a JSON object, returning fallback on any failure."""

    return call_llm_json_result(messages, fallback).content


def call_llm_json_result(messages: list[dict[str, str]], fallback: Any) -> LLMCallResult:
    """Call OpenRouter, parse JSON, and return structured diagnostics."""

    raw_result = _call_openrouter_result(messages)
    if not raw_result.ok:
        result = _with_fallback(raw_result, fallback)
        _record_call_result(result)
        return result

    try:
        parsed = json.loads(_strip_json_fence(str(raw_result.content)))
    except json.JSONDecodeError as exc:
        message = f"LLM response was received but JSON parsing failed: {exc.msg}"
        _add_warning(f"{message}; deterministic fallback was used.")
        result = LLMCallResult(
            ok=False,
            content=deepcopy(fallback),
            model=raw_result.model,
            used_fallback=True,
            error_type="json_validation_error",
            error_message=message,
        )
        _record_call_result(result)
        return result

    result = LLMCallResult(
        ok=True,
        content=parsed,
        model=raw_result.model,
        used_fallback=False,
    )
    _record_call_result(result)
    return result


def call_llm_text(messages: list[dict[str, str]], fallback: str) -> str:
    """Call OpenRouter and return text, returning fallback on any failure."""

    return str(call_llm_text_result(messages, fallback).content)


def call_llm_text_result(messages: list[dict[str, str]], fallback: str) -> LLMCallResult:
    """Call OpenRouter for text and return structured diagnostics."""

    raw_result = _call_openrouter_result(messages)
    if not raw_result.ok:
        result = _with_fallback(raw_result, fallback)
        _record_call_result(result)
        return result

    result = LLMCallResult(
        ok=True,
        content=str(raw_result.content),
        model=raw_result.model,
        used_fallback=False,
    )
    _record_call_result(result)
    return result


def test_llm_connection() -> LLMCallResult:
    """Send a tiny JSON request to verify OpenRouter connectivity."""

    result = call_llm_json_result(
        [
            {
                "role": "system",
                "content": "Return strict JSON only. Do not include markdown.",
            },
            {
                "role": "user",
                "content": 'Return exactly JSON: {"status": "ok"}',
            },
        ],
        fallback={"status": "fallback"},
    )
    if not result.ok:
        return result
    if not isinstance(result.content, dict) or result.content.get("status") != "ok":
        message = "LLM connection response did not contain {'status': 'ok'}."
        checked = LLMCallResult(
            ok=False,
            content={"status": "fallback"},
            model=result.model,
            used_fallback=True,
            error_type="json_validation_error",
            error_message=message,
        )
        _record_call_result(checked)
        return checked
    return result


def _refresh_status() -> None:
    _load_project_dotenv()
    api_key = _env_value("OPENROUTER_API_KEY")
    _STATUS.available = bool(api_key)
    _STATUS.model = _env_value("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    if not _STATUS.model:
        _STATUS.model = DEFAULT_OPENROUTER_MODEL
    _STATUS.key_preview = _mask_key(api_key)
    _STATUS.site_url = _env_value("OPENROUTER_SITE_URL")
    _STATUS.app_name = _env_value("OPENROUTER_APP_NAME")
    _STATUS.dotenv_path = str(DOTENV_PATH)

    if not api_key:
        _add_warning(MISSING_KEY_WARNING)
    elif MISSING_KEY_WARNING in _STATUS.warnings:
        _STATUS.warnings.remove(MISSING_KEY_WARNING)


def _call_openrouter_result(messages: list[dict[str, str]]) -> LLMCallResult:
    _refresh_status()
    api_key = _env_value("OPENROUTER_API_KEY")
    model = _STATUS.model or DEFAULT_OPENROUTER_MODEL
    if not api_key:
        return LLMCallResult(
            ok=False,
            model=model,
            used_fallback=True,
            error_type="missing_api_key",
            error_message="OpenRouter API key is missing.",
        )

    endpoint = _openrouter_endpoint()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    site_url = _env_value("OPENROUTER_SITE_URL")
    app_name = _env_value("OPENROUTER_APP_NAME")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        message = _safe_error_message(str(exc), api_key)
        _add_warning(f"OpenRouter network error; deterministic fallback was used. {message}")
        return LLMCallResult(
            ok=False,
            model=model,
            used_fallback=True,
            error_type="network_error",
            error_message=message,
        )

    if response.status_code == 429:
        message = _safe_response_text(response, api_key)
        _add_warning("OpenRouter rate limit reached; deterministic fallback was used.")
        return LLMCallResult(
            ok=False,
            model=model,
            used_fallback=True,
            error_type="rate_limit",
            error_message=message or "OpenRouter rate limit reached.",
            http_status=response.status_code,
        )
    if response.status_code >= 400:
        message = _safe_response_text(response, api_key)
        _add_warning(
            f"OpenRouter request failed with status {response.status_code}; "
            "deterministic fallback was used."
        )
        return LLMCallResult(
            ok=False,
            model=model,
            used_fallback=True,
            error_type="http_error",
            error_message=message or f"OpenRouter request failed with status {response.status_code}.",
            http_status=response.status_code,
        )

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        message = _safe_response_text(response, api_key)
        _add_warning("OpenRouter returned an unexpected response; deterministic fallback was used.")
        return LLMCallResult(
            ok=False,
            model=model,
            used_fallback=True,
            error_type="api_response_error",
            error_message=message or "OpenRouter returned an unexpected response shape.",
            http_status=response.status_code,
        )

    if not isinstance(content, str) or not content.strip():
        _add_warning("OpenRouter returned empty content; deterministic fallback was used.")
        return LLMCallResult(
            ok=False,
            model=model,
            used_fallback=True,
            error_type="empty_content",
            error_message="OpenRouter returned empty content.",
            http_status=response.status_code,
        )

    return LLMCallResult(
        ok=True,
        content=content.strip(),
        model=model,
        used_fallback=False,
        http_status=response.status_code,
    )


def _openrouter_endpoint() -> str:
    base_url = _env_value("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
    if not base_url:
        base_url = DEFAULT_OPENROUTER_BASE_URL
    return base_url.rstrip("/") + "/chat/completions"


def _load_project_dotenv() -> None:
    load_dotenv(DOTENV_PATH, override=False)


def _env_value(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None and value.strip():
        return value.strip()

    if DOTENV_PATH.exists():
        file_value = dotenv_values(DOTENV_PATH).get(name)
        if file_value is not None and str(file_value).strip():
            return str(file_value).strip()

    return default


def _mask_key(api_key: str) -> str:
    if not api_key:
        return ""
    prefix = api_key[:8]
    return f"{prefix}..."


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _with_fallback(result: LLMCallResult, fallback: Any) -> LLMCallResult:
    return LLMCallResult(
        ok=False,
        content=deepcopy(fallback),
        model=result.model,
        used_fallback=True,
        error_type=result.error_type,
        error_message=result.error_message,
        http_status=result.http_status,
    )


def _record_call_result(result: LLMCallResult) -> None:
    global _LAST_CALL_RESULT
    _LAST_CALL_RESULT = deepcopy(result)
    _CALL_HISTORY.append(deepcopy(result))


def _safe_response_text(response: requests.Response, api_key: str) -> str:
    return _safe_error_message(getattr(response, "text", ""), api_key)


def _safe_error_message(message: str, api_key: str = "") -> str:
    if not message:
        return ""
    safe = str(message)
    if api_key:
        safe = safe.replace(api_key, "[redacted]")
    safe = " ".join(safe.split())
    return safe[:500]


def _add_warning(message: str) -> None:
    if message not in _STATUS.warnings:
        _STATUS.warnings.append(message)
