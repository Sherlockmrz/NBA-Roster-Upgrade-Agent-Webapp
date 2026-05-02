from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from components import merge_parsed_with_sidebar, parsed_query_to_session_updates


def test_sidebar_follows_parsed_query_when_manual_override_off():
    parsed = {
        "team": "Golden State Warriors",
        "goal": "interior defense",
        "top_k": 5,
        "recent_games": 10,
        "min_games": 15,
        "min_avg_minutes": 20,
        "exclude_current_team": True,
        "ranking_mode": "Best Talent",
    }
    sidebar = {
        "team": "Boston Celtics",
        "goal": "shooting",
        "top_k": 3,
        "recent_games": 5,
        "min_games": 30,
        "min_avg_minutes": 10,
        "exclude_current_team": False,
        "ranking_mode": "Hidden Gems",
    }

    final = merge_parsed_with_sidebar(parsed, sidebar, use_sidebar_override=False)
    updates = parsed_query_to_session_updates(final)

    assert final["team"] == "Golden State Warriors"
    assert final["min_avg_minutes"] == 20
    assert updates["selected_team"] == "Golden State Warriors"
    assert updates["selected_min_avg_minutes"] == 20


def test_sidebar_override_wins_only_when_enabled():
    parsed = {
        "team": "Golden State Warriors",
        "goal": "interior defense",
        "top_k": 5,
        "recent_games": 10,
        "min_games": 15,
        "min_avg_minutes": 20,
        "exclude_current_team": True,
        "ranking_mode": "Best Talent",
    }
    sidebar = {
        "team": "Boston Celtics",
        "goal": "shooting",
        "top_k": 3,
        "recent_games": 5,
        "min_games": 30,
        "min_avg_minutes": 10,
        "exclude_current_team": False,
        "ranking_mode": "Hidden Gems",
    }

    final = merge_parsed_with_sidebar(parsed, sidebar, use_sidebar_override=True)

    assert final["team"] == "Boston Celtics"
    assert final["top_k"] == 3
    assert final["min_avg_minutes"] == 10
    assert final["ranking_mode"] == "Hidden Gems"
