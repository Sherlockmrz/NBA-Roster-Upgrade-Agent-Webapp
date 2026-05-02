from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_agent.agent import run_roster_agent
from nba_agent.data.loader import load_raw_data
from nba_agent.data.preprocessing import prepare_data
from nba_agent.llm import client
from nba_agent.llm import parser
from nba_agent.llm.parser import normalize_team_value, parse_user_query


DATA_DIR = Path("data/raw")


def test_parse_user_query_missing_key_uses_deterministic_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "DOTENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    prepared = prepare_data(load_raw_data(DATA_DIR))
    query = (
        "Recommend top 5 players for the Warriors to improve interior defense "
        "using the last 10 games. Only include players with at least 20 games "
        "and 18 average minutes."
    )

    parsed = parse_user_query(query, defaults={}, teams_df=prepared.teams, use_llm=True)

    assert parsed["team"] == "Golden State Warriors"
    assert parsed["goal"] == "interior defense"
    assert parsed["top_k"] == 5
    assert parsed["recent_games"] == 10
    assert parsed["min_games"] == 20
    assert parsed["min_avg_minutes"] == 18
    assert parsed["exclude_current_team"] is True
    assert parsed["ranking_mode"] == "Best Talent"
    assert parsed["unavailable_constraints"] == []


def test_parse_user_query_preserves_unavailable_salary_constraint(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "DOTENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    prepared = prepare_data(load_raw_data(DATA_DIR))

    parsed = parse_user_query(
        "Recommend top 3 players for the Lakers using salary constraints.",
        defaults={},
        teams_df=prepared.teams,
        use_llm=True,
    )

    assert "salary" in parsed["unavailable_constraints"]


def test_run_roster_agent_use_llm_without_key_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "DOTENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    result = run_roster_agent(
        "Recommend top 2 players for the Lakers to improve rebounding.",
        {"min_games": 15, "min_avg_minutes": 15},
        data_dir="data/raw",
        use_llm=True,
    )

    assert result.parsed_query.team_name == "Los Angeles Lakers"
    assert result.parsed_query.goal == "rebounding"
    assert len(result.ranked_df) == 2
    assert any("API key is missing" in warning for warning in result.warnings)


def test_natural_language_query_overrides_sidebar_defaults():
    result = run_roster_agent(
        (
            "Recommend top 5 players for the Warriors to improve interior defense "
            "using the last 10 games. Only include players with at least 20 games "
            "and 18 average minutes."
        ),
        {
            "team": "Golden State Warriors",
            "goal": "interior defense",
            "top_k": 5,
            "recent_games": 10,
            "min_games": 15,
            "min_avg_minutes": 15,
            "exclude_current_team": True,
            "ranking_mode": "Best Talent",
        },
        data_dir="data/raw",
        use_llm=False,
    )

    assert result.parsed_query.min_games == 20
    assert result.parsed_query.min_avg_minutes == 18


def test_team_normalization_accepts_warriors_and_gsw():
    prepared = prepare_data(load_raw_data(DATA_DIR))

    assert normalize_team_value("Warriors", prepared.teams) == "Golden State Warriors"
    assert normalize_team_value("GSW", prepared.teams) == "Golden State Warriors"


def test_invalid_llm_team_does_not_override_query_team(monkeypatch):
    prepared = prepare_data(load_raw_data(DATA_DIR))

    def fake_call_llm_json(messages, fallback):
        return {
            "team": ";",
            "goal": "interior defense",
            "top_k": 5,
            "recent_games": 10,
            "min_games": 15,
            "min_avg_minutes": 20,
            "exclude_current_team": True,
            "ranking_mode": "Best Talent",
            "unavailable_constraints": [],
        }

    monkeypatch.setattr(parser, "call_llm_json", fake_call_llm_json)

    parsed = parse_user_query(
        (
            "Recommend top 5 players for the Warriors to improve interior defense "
            "using the last 10 games. Only include players with at least 15 games "
            "and 20 average minutes."
        ),
        defaults={"team": "Boston Celtics"},
        teams_df=prepared.teams,
        use_llm=True,
    )

    assert parsed["team"] == "Golden State Warriors"
    assert parsed["top_k"] == 5
    assert parsed["recent_games"] == 10
    assert parsed["min_games"] == 15
    assert parsed["min_avg_minutes"] == 20


def test_query_top_last_games_and_minutes_are_parsed():
    prepared = prepare_data(load_raw_data(DATA_DIR))

    parsed = parse_user_query(
        (
            "Recommend top five players for GSW to improve interior defense using "
            "the last 10 games. Only include players with at least 15 games and "
            "20 avg minutes."
        ),
        defaults={},
        teams_df=prepared.teams,
        use_llm=False,
    )

    assert parsed["team"] == "Golden State Warriors"
    assert parsed["top_k"] == 5
    assert parsed["recent_games"] == 10
    assert parsed["min_games"] == 15
    assert parsed["min_avg_minutes"] == 20
