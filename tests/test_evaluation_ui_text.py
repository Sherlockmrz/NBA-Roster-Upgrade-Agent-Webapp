from pathlib import Path


APP_SOURCE = Path(__file__).resolve().parents[1] / "app" / "app.py"


def test_evaluation_page_contains_fairness_note():
    source = APP_SOURCE.read_text()

    assert "Important fairness note" in source
    assert "not independent ground-truth measures" in source


def test_evaluation_page_splits_metric_groups():
    source = APP_SOURCE.read_text()

    assert "Fair Audit Metrics" in source
    assert "Internal Objective Metrics" in source
    assert "How to read this evaluation fairly" in source
