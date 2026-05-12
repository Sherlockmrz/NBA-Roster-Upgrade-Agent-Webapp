from pathlib import Path


APP_SOURCE = Path(__file__).resolve().parents[1] / "app" / "app.py"


def test_app_does_not_display_full_api_key_or_key_preview():
    source = APP_SOURCE.read_text()

    assert "key_preview" not in source
    assert "Key preview" not in source
    assert "OPENROUTER_API_KEY" not in source


def test_static_pre_run_workflow_strip_is_not_rendered():
    source = APP_SOURCE.read_text()

    assert "render_workflow_strip(" not in source
    assert "Full Agent Pipeline" in source
    assert "LLM Tool Selection Decision" in source


def test_hero_subtitle_does_not_include_feasibility_critique():
    source = APP_SOURCE.read_text().lower()

    assert "feasibility critique" not in source


def test_active_app_does_not_show_placeholder_card_fields_or_ranking_modes():
    source = APP_SOURCE.read_text()

    assert "Player image placeholder" not in source
    assert "Salary: unavailable in current dataset" not in source
    assert "Salary data is unavailable in the current dataset." not in source
    assert 'st.radio("Ranking mode"' not in source
    assert "Realistic Fit" not in source
    assert "Hidden Gems" not in source
    assert 'if "ranking mode" in str(item).lower()' in source
