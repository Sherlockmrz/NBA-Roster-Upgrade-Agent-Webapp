"""Tool B: deterministic player strength representation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_agent.schemas import PreparedNBAData
from nba_agent.tools.need_diagnosis import CORE_METRICS


RADAR_METRICS = ["PTS", "AST", "REB", "STL", "BLK", "FG3_PCT"]


RADAR_LABELS = {
    "PTS": "Scoring",
    "AST": "Playmaking",
    "REB": "Rebounding",
    "STL": "Perimeter Defense",
    "BLK": "Rim Protection",
    "FG3_PCT": "Three-Point Shooting",
}


def build_player_strengths(
    prepared_data: PreparedNBAData,
    season: int | None = None,
    min_games: int = 15,
    min_avg_minutes: float = 15.0,
    exclude_current_team: bool = False,
    current_team_id: int | None = None,
) -> pd.DataFrame:
    """Return player strength vectors using existing box-score features."""

    season = prepared_data.default_season if season is None else season
    season_df = prepared_data.player_game_df[
        prepared_data.player_game_df["SEASON"] == season
    ].copy()

    player_summary = (
        season_df.groupby(["PLAYER_ID", "PLAYER_NAME", "TEAM_ID"], as_index=False)
        .agg(
            {
                "GAME_ID": pd.Series.nunique,
                "MIN_FLOAT": "mean",
                "PTS": "mean",
                "REB": "mean",
                "AST": "mean",
                "STL": "mean",
                "BLK": "mean",
                "FG3M": "mean",
                "FG3A": "mean",
            }
        )
        .rename(columns={"GAME_ID": "GP", "MIN_FLOAT": "AVG_MIN"})
    )

    player_summary["FG3_PCT"] = np.where(
        player_summary["FG3A"] > 0,
        player_summary["FG3M"] / player_summary["FG3A"],
        0,
    )

    player_summary = player_summary[
        (player_summary["GP"] >= min_games)
        & (player_summary["AVG_MIN"] >= min_avg_minutes)
    ].copy()

    if exclude_current_team:
        if current_team_id is None:
            raise ValueError(
                "current_team_id is required when exclude_current_team=True."
            )
        player_summary = player_summary[
            player_summary["TEAM_ID"] != current_team_id
        ].copy()

    if player_summary.empty:
        player_summary["CURRENT_TEAM"] = pd.Series(dtype="object")
        return player_summary

    core_means = player_summary[CORE_METRICS].mean()
    core_stds = player_summary[CORE_METRICS].std(ddof=0).replace(0, 1)
    z_strength = ((player_summary[CORE_METRICS] - core_means) / core_stds).clip(
        lower=0
    )

    for metric in CORE_METRICS:
        player_summary[f"{metric}_strength"] = z_strength[metric]

    for metric in RADAR_METRICS:
        player_summary[f"{metric}_radar"] = player_summary[metric].rank(pct=True) * 100

    player_summary["CURRENT_TEAM"] = player_summary["TEAM_ID"].map(
        prepared_data.team_name_map
    )

    return player_summary.reset_index(drop=True)


def tool_b_player_strengths(*args, **kwargs) -> pd.DataFrame:
    """Notebook-compatible name for Tool B."""

    return build_player_strengths(*args, **kwargs)
