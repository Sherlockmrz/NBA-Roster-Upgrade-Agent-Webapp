"""Tool A: deterministic team need diagnosis."""

from __future__ import annotations

import pandas as pd

from nba_agent.schemas import PreparedNBAData


CORE_METRICS = ["REB", "AST", "STL", "BLK", "FG3_PCT"]


CORE_METRIC_LABELS = {
    "REB": "Rebounding",
    "AST": "Playmaking",
    "STL": "Perimeter Defense",
    "BLK": "Rim Protection",
    "FG3_PCT": "Three-Point Shooting",
}


GOAL_MAP = {
    "interior defense": ["BLK", "REB"],
    "rim protection": ["BLK"],
    "rebounding": ["REB"],
    "playmaking": ["AST"],
    "perimeter defense": ["STL"],
    "three-point shooting": ["FG3_PCT"],
    "shooting": ["FG3_PCT"],
}


NEED_COLUMNS = [
    "metric",
    "label",
    "team_value",
    "league_mean",
    "z_score",
    "need_weight",
    "goal_boosted",
]


def apply_goal_boost(
    need_df: pd.DataFrame, goal_text: str | None = None, boost: float = 1.5
) -> pd.DataFrame:
    """Boost need weights for a user-stated basketball goal."""

    out = need_df.copy()
    out["goal_boosted"] = False

    if not goal_text:
        return out[NEED_COLUMNS].sort_values(
            "need_weight", ascending=False
        ).reset_index(drop=True)

    goal_lower = goal_text.lower().strip()
    boosted_metrics: set[str] = set()

    for key, metrics in GOAL_MAP.items():
        if key in goal_lower:
            boosted_metrics.update(metrics)

    if boosted_metrics:
        mask = out["metric"].isin(boosted_metrics)
        out.loc[mask, "need_weight"] = out.loc[mask, "need_weight"] * boost
        out.loc[mask, "goal_boosted"] = True

    return out[NEED_COLUMNS].sort_values("need_weight", ascending=False).reset_index(
        drop=True
    )


def diagnose_team_needs(
    prepared_data: PreparedNBAData,
    team_id: int,
    season: int | None = None,
    recent_games: int = 10,
    goal: str | None = None,
    goal_boost: float = 1.5,
) -> pd.DataFrame:
    """Return Tool A need weights from recent team box-score z-scores."""

    season = prepared_data.default_season if season is None else season
    team_game_df = prepared_data.team_game_df

    team_part = team_game_df[
        (team_game_df["TEAM_ID"] == team_id) & (team_game_df["SEASON"] == season)
    ].sort_values("GAME_DATE_EST")

    if team_part.empty:
        raise ValueError(f"No team games found for team_id={team_id}, season={season}.")

    team_recent = team_part.tail(recent_games)

    league_recent = (
        team_game_df[team_game_df["SEASON"] == season]
        .sort_values("GAME_DATE_EST")
        .groupby("TEAM_ID", as_index=False)
        .tail(recent_games)
        .groupby("TEAM_ID", as_index=False)[CORE_METRICS]
        .mean()
    )

    team_values = team_recent[CORE_METRICS].mean()
    league_mean = league_recent[CORE_METRICS].mean()
    league_std = league_recent[CORE_METRICS].std(ddof=0).replace(0, 1)

    z_scores = (team_values - league_mean) / league_std
    need_weights = (-z_scores).clip(lower=0)

    need_df = pd.DataFrame(
        {
            "metric": CORE_METRICS,
            "label": [CORE_METRIC_LABELS[metric] for metric in CORE_METRICS],
            "team_value": [team_values[metric] for metric in CORE_METRICS],
            "league_mean": [league_mean[metric] for metric in CORE_METRICS],
            "z_score": [z_scores[metric] for metric in CORE_METRICS],
            "need_weight": [need_weights[metric] for metric in CORE_METRICS],
            "goal_boosted": False,
        }
    )

    return apply_goal_boost(need_df, goal_text=goal, boost=goal_boost)


def tool_a_team_need_diagnosis(*args, **kwargs) -> pd.DataFrame:
    """Notebook-compatible name for Tool A."""

    return diagnose_team_needs(*args, **kwargs)
