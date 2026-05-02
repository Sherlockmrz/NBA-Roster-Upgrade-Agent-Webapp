from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_agent.data.loader import load_raw_data
from nba_agent.data.preprocessing import find_team_id, parse_minutes, prepare_data
from nba_agent.tools.fit_ranking import RANKING_COLUMNS, rank_players_by_fit
from nba_agent.tools.need_diagnosis import NEED_COLUMNS, diagnose_team_needs
from nba_agent.tools.player_strength import build_player_strengths


DATA_DIR = Path("data/raw")


def test_data_loading():
    raw = load_raw_data(DATA_DIR)

    assert not raw.teams.empty
    assert not raw.games.empty
    assert not raw.games_details.empty
    assert {"TEAM_ID", "NICKNAME"}.issubset(raw.teams.columns)
    assert {"GAME_ID", "SEASON"}.issubset(raw.games.columns)
    assert {"PLAYER_NAME", "MIN"}.issubset(raw.games_details.columns)


def test_parse_minutes():
    assert parse_minutes("18:06") == 18.1
    assert parse_minutes("31") == 31.0
    assert parse_minutes("") == 0.0
    assert parse_minutes(None) == 0.0
    assert parse_minutes("not-a-minute") == 0.0


def test_tool_a_output_columns():
    prepared = prepare_data(load_raw_data(DATA_DIR))
    team_id = find_team_id("Warriors", prepared.team_lookup)

    need_df = diagnose_team_needs(
        prepared,
        team_id=team_id,
        recent_games=10,
        goal="interior defense",
    )

    assert list(need_df.columns) == NEED_COLUMNS
    assert len(need_df) > 0
    assert pd.api.types.is_numeric_dtype(need_df["need_weight"])
    assert need_df["goal_boosted"].isin([True, False]).all()


def test_tool_b_candidate_filtering():
    prepared = prepare_data(load_raw_data(DATA_DIR))
    team_id = find_team_id("Warriors", prepared.team_lookup)

    player_df = build_player_strengths(
        prepared,
        min_games=20,
        min_avg_minutes=18,
        exclude_current_team=True,
        current_team_id=team_id,
    )

    assert not player_df.empty
    assert (player_df["GP"] >= 20).all()
    assert (player_df["AVG_MIN"] >= 18).all()
    assert (player_df["TEAM_ID"] != team_id).all()


def test_tool_c_returns_top_k_ranking():
    prepared = prepare_data(load_raw_data(DATA_DIR))
    team_id = find_team_id("Warriors", prepared.team_lookup)
    need_df = diagnose_team_needs(
        prepared,
        team_id=team_id,
        recent_games=10,
        goal="interior defense",
    )
    player_df = build_player_strengths(
        prepared,
        min_games=20,
        min_avg_minutes=18,
    )

    ranked_df = rank_players_by_fit(
        need_df,
        player_df,
        team_id=team_id,
        top_k=5,
        exclude_current_team=True,
    )

    assert list(ranked_df.columns) == RANKING_COLUMNS
    assert len(ranked_df) == 5
    assert ranked_df["fit_score"].is_monotonic_decreasing
