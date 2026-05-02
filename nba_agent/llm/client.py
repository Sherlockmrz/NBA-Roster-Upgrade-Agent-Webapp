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


_STATUS = LLMStatus()


def get_llm_status() -> LLMStatus:
    """Return a copy of the most recent LLM status."""

    _refresh_status()
    return deepcopy(_STATUS)


def is_llm_available() -> bool:
    """Return whether an OpenRouter API key is configured."""

    return get_llm_status().available


def call_llm_json(messages: list[dict[str, str]], fallback: Any) -> Any:
    """Call OpenRouter and parse a JSON object, returning fallback on any failure."""

    text = _call_openrouter(messages, json_mode=True)
    if text is None:
        return fallback

    try:
        return json.loads(_strip_json_fence(text))
    except json.JSONDecodeError:
        _add_warning("OpenRouter returned invalid JSON; deterministic fallback was used.")
        return fallback


def call_llm_text(messages: list[dict[str, str]], fallback: str) -> str:
    """Call OpenRouter and return text, returning fallback on any failure."""

    text = _call_openrouter(messages, json_mode=False)
    if text is None:
        return fallback
    return text


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


def _call_openrouter(messages: list[dict[str, str]], json_mode: bool = False) -> str | None:
    _refresh_status()
    api_key = _env_value("OPENROUTER_API_KEY")
    if not api_key:
        return None

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
        "model": _STATUS.model,
        "messages": messages,
        "temperature": 0,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        _add_warning(f"OpenRouter network error; deterministic fallback was used. {exc}")
        return None

    if response.status_code == 429:
        _add_warning("OpenRouter rate limit reached; deterministic fallback was used.")
        return None
    if response.status_code >= 400:
        _add_warning(
            f"OpenRouter request failed with status {response.status_code}; "
            "deterministic fallback was used."
        )
        return None

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        _add_warning("OpenRouter returned an unexpected response; deterministic fallback was used.")
        return None

    if not isinstance(content, str) or not content.strip():
        _add_warning("OpenRouter returned empty content; deterministic fallback was used.")
        return None

    return content.strip()


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


def _add_warning(message: str) -> None:
    if message not in _STATUS.warnings:
        _STATUS.warnings.append(message)
