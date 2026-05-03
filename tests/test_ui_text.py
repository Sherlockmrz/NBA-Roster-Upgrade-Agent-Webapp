from pathlib import Path


APP_SOURCE = Path(__file__).resolve().parents[1] / "app" / "app.py"


def test_app_does_not_display_full_api_key_or_key_preview():
    source = APP_SOURCE.read_text()

    assert "key_preview" not in source
    assert "Key preview" not in source
    assert "OPENROUTER_API_KEY" not in source


def test_workflow_strip_does_not_include_critique():
    source = APP_SOURCE.read_text()
    workflow_call = source.rsplit("render_workflow_strip(", 1)[1].split(")", 1)[0]

    assert "Critique" not in workflow_call


def test_hero_subtitle_does_not_include_feasibility_critique():
    source = APP_SOURCE.read_text().lower()

    assert "feasibility critique" not in source
