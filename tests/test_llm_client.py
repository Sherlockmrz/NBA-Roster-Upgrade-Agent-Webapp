from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_agent.llm import client


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_missing_openrouter_key_returns_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "DOTENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    json_fallback = {"mode": "fallback"}
    text_fallback = "fallback text"

    assert client.call_llm_json([{"role": "user", "content": "Return JSON."}], json_fallback) == json_fallback
    assert client.call_llm_text([{"role": "user", "content": "Return text."}], text_fallback) == text_fallback

    status = client.get_llm_status()
    assert status.available is False
    assert status.model == "openrouter/free"
    assert any("API key is missing" in warning for warning in status.warnings)


def test_missing_openrouter_key_returns_structured_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "DOTENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    fallback = {"mode": "fallback"}
    result = client.call_llm_json_result(
        [{"role": "user", "content": "Return JSON."}],
        fallback,
    )

    assert result.ok is False
    assert result.content == fallback
    assert result.used_fallback is True
    assert result.error_type == "missing_api_key"


def test_api_call_failure_returns_safe_error_status(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "DOTENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-secret-value")
    monkeypatch.setenv("OPENROUTER_MODEL", "test/model")

    def fake_post(*args, **kwargs):
        return _FakeResponse(
            status_code=401,
            text="unauthorized for sk-test-secret-value",
        )

    monkeypatch.setattr(client.requests, "post", fake_post)

    result = client.call_llm_text_result(
        [{"role": "user", "content": "Hello"}],
        "fallback",
    )

    assert result.ok is False
    assert result.content == "fallback"
    assert result.used_fallback is True
    assert result.error_type == "http_error"
    assert result.http_status == 401
    assert "sk-test-secret-value" not in result.error_message
    assert "[redacted]" in result.error_message


def test_invalid_json_returns_validation_failure_status(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "DOTENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-secret-value")
    monkeypatch.setenv("OPENROUTER_MODEL", "test/model")

    def fake_post(*args, **kwargs):
        return _FakeResponse(
            status_code=200,
            payload={"choices": [{"message": {"content": "not json"}}]},
        )

    monkeypatch.setattr(client.requests, "post", fake_post)

    fallback = {"mode": "fallback"}
    result = client.call_llm_json_result(
        [{"role": "user", "content": "Return JSON."}],
        fallback,
    )

    assert result.ok is False
    assert result.content == fallback
    assert result.used_fallback is True
    assert result.error_type == "json_validation_error"
    assert "JSON parsing failed" in result.error_message


def test_openrouter_request_uses_expected_headers_and_body(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "DOTENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-secret-value")
    monkeypatch.setenv("OPENROUTER_MODEL", "test/model:free")
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://example.test")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "Test App")
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse(
            status_code=200,
            payload={"choices": [{"message": {"content": '{"status": "ok"}'}}]},
        )

    monkeypatch.setattr(client.requests, "post", fake_post)

    result = client.call_llm_json_result(
        [{"role": "user", "content": "Return JSON."}],
        {"status": "fallback"},
    )

    assert result.ok is True
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test-secret-value"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["headers"]["HTTP-Referer"] == "https://example.test"
    assert captured["headers"]["X-Title"] == "Test App"
    assert captured["json"]["model"] == "test/model:free"
    assert captured["json"]["messages"] == [{"role": "user", "content": "Return JSON."}]
    assert captured["json"]["temperature"] == 0.2
    assert "response_format" not in captured["json"]


def test_root_dotenv_key_is_detected_and_masked(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "OPENROUTER_API_KEY=sk-test-secret-value\n"
        "OPENROUTER_MODEL=test/model\n"
        "OPENROUTER_SITE_URL=https://example.test\n"
        "OPENROUTER_APP_NAME=Test App\n"
    )
    monkeypatch.setattr(client, "DOTENV_PATH", dotenv_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_SITE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_APP_NAME", raising=False)

    status = client.get_llm_status()

    assert status.available is True
    assert status.model == "test/model"
    assert status.key_preview == "sk-test-..."
    assert "secret-value" not in status.key_preview
    assert status.site_url == "https://example.test"
    assert status.app_name == "Test App"
