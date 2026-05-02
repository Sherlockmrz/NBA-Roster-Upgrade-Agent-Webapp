from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_agent.llm import client


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
