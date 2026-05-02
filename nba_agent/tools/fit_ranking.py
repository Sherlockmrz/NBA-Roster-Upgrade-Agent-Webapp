"""Tool C: deterministic fit scoring and ranking."""

from __future__ import annotations

import pandas as pd

from nba_agent.tools.need_diagnosis import CORE_METRIC_LABELS, CORE_METRICS


RANKING_COLUMNS = [
    "PLAYER_NAME",
    "CURRENT_TEAM",
    "GP",
    "AVG_MIN",
    "fit_score",
    "best_match",
]


def rank_players_by_fit(
    need_df: pd.DataFrame,
    player_strength_df: pd.DataFrame,
    team_id: int | None = None,
    top_k: int = 5,
    exclude_current_team: bool = True,
) -> pd.DataFrame:
    """Rank candidate players by weighted match to Tool A need weights."""

    candidates = player_strength_df.copy()

    if exclude_current_team:
        if team_id is None:
            raise ValueError("team_id is required when exclude_current_team=True.")
        candidates = candidates[candidates["TEAM_ID"] != team_id].copy()

    if candidates.empty:
        return pd.DataFrame(columns=RANKING_COLUMNS)

    weight_column = (
        "adjusted_need_weight"
        if "adjusted_need_weight" in need_df.columns
        else "need_weight"
    )
    need_map = dict(zip(need_df["metric"], need_df[weight_column]))

    candidates["fit_score"] = 0.0
    for metric in CORE_METRICS:
        strength_column = f"{metric}_strength"
        if strength_column not in candidates.columns:
            raise ValueError(f"Missing player strength column: {strength_column}")
        candidates["fit_score"] += (
            need_map.get(metric, 0) * candidates[strength_column]
        )

    def best_match(row: pd.Series) -> str:
        parts = {}
        for metric in CORE_METRICS:
            parts[CORE_METRIC_LABELS[metric]] = (
                need_map.get(metric, 0) * row[f"{metric}_strength"]
            )

        best_two = sorted(parts.items(), key=lambda item: item[1], reverse=True)[:2]
        labels = [label for label, score in best_two if score > 0]
        return " + ".join(labels) if labels else "General fit"

    candidates["best_match"] = candidates.apply(best_match, axis=1)

    ranked = candidates.sort_values("fit_score", ascending=False).reset_index(drop=True)
    return ranked.head(top_k)[RANKING_COLUMNS]


def tool_c_rank_players(*args, **kwargs) -> pd.DataFrame:
    """Notebook-compatible name for Tool C."""

    return rank_players_by_fit(*args, **kwargs)
