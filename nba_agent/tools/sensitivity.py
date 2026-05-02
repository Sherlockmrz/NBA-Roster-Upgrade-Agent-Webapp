"""Sensitivity / robustness analysis for deterministic fit rankings."""

from __future__ import annotations

import pandas as pd

from nba_agent.schemas import SensitivityResult
from nba_agent.tools.fit_ranking import rank_players_by_fit


def run_sensitivity_check(
    need_df: pd.DataFrame,
    player_strength_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    team_id: int | None = None,
    top_k: int = 5,
    perturbation: float = 0.10,
) -> SensitivityResult:
    """Rerank after a small deterministic need-weight perturbation."""

    original_top_players = _top_players(ranked_df, top_k)
    if ranked_df.empty or player_strength_df.empty or need_df.empty:
        return SensitivityResult(
            stability_label="Unstable",
            top_k_overlap=0.0,
            original_top_players=original_top_players,
            perturbed_top_players=[],
            explanation="Robustness could not be established because one or more ranking inputs were empty.",
            rank_comparison_df=_rank_comparison(original_top_players, []),
        )

    perturbed_need_df = _perturb_need_weights(need_df, perturbation=perturbation)
    perturbed_ranked_df = rank_players_by_fit(
        perturbed_need_df,
        player_strength_df,
        team_id=team_id,
        top_k=top_k,
        exclude_current_team=False,
    )
    perturbed_top_players = _top_players(perturbed_ranked_df, top_k)

    overlap_count = len(set(original_top_players) & set(perturbed_top_players))
    denominator = max(len(original_top_players), 1)
    top_k_overlap = overlap_count / denominator
    stability_label = _stability_label(top_k_overlap)

    return SensitivityResult(
        stability_label=stability_label,
        top_k_overlap=top_k_overlap,
        original_top_players=original_top_players,
        perturbed_top_players=perturbed_top_players,
        explanation=_explanation(stability_label, top_k_overlap, perturbation),
        rank_comparison_df=_rank_comparison(original_top_players, perturbed_top_players),
    )


def _perturb_need_weights(need_df: pd.DataFrame, perturbation: float) -> pd.DataFrame:
    """Apply an interpretable alternating +/- perturbation to the ranking weights."""

    out = need_df.copy()
    base_column = (
        "adjusted_need_weight"
        if "adjusted_need_weight" in out.columns
        else "need_weight"
    )
    factors = [
        1 + perturbation if index % 2 == 0 else 1 - perturbation
        for index in range(len(out))
    ]
    out["adjusted_need_weight"] = out[base_column] * factors
    return out


def _top_players(ranked_df: pd.DataFrame, top_k: int) -> list[str]:
    if ranked_df.empty or "PLAYER_NAME" not in ranked_df.columns:
        return []
    return ranked_df["PLAYER_NAME"].head(top_k).astype(str).tolist()


def _rank_comparison(
    original_top_players: list[str], perturbed_top_players: list[str]
) -> pd.DataFrame:
    players = list(dict.fromkeys(original_top_players + perturbed_top_players))
    rows = []
    for player in players:
        original_rank = (
            original_top_players.index(player) + 1
            if player in original_top_players
            else None
        )
        perturbed_rank = (
            perturbed_top_players.index(player) + 1
            if player in perturbed_top_players
            else None
        )
        rows.append(
            {
                "PLAYER_NAME": player,
                "original_rank": original_rank,
                "perturbed_rank": perturbed_rank,
                "in_both": player in original_top_players
                and player in perturbed_top_players,
            }
        )
    return pd.DataFrame(rows)


def _stability_label(top_k_overlap: float) -> str:
    if top_k_overlap >= 0.8:
        return "Stable"
    if top_k_overlap >= 0.4:
        return "Somewhat Stable"
    return "Unstable"


def _explanation(label: str, overlap: float, perturbation: float) -> str:
    percent = round(overlap * 100)
    perturb_percent = round(perturbation * 100)
    if label == "Stable":
        return (
            f"The recommendation set is stable: {percent}% of the original top players "
            f"remain after a +/-{perturb_percent}% need-weight perturbation."
        )
    if label == "Somewhat Stable":
        return (
            f"The recommendation set is somewhat stable: {percent}% of the original top players "
            f"remain after a +/-{perturb_percent}% need-weight perturbation."
        )
    return (
        f"The recommendation set is unstable: only {percent}% of the original top players "
        f"remain after a +/-{perturb_percent}% need-weight perturbation."
    )
