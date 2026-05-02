"""Need-weight reasoning layer between Tool A and Tool C."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from nba_agent.llm.client import call_llm_json
from nba_agent.llm.prompts import NEED_REASONER_PROMPT
from nba_agent.schemas import AnalysisRequest, NeedReasoningResult


FALLBACK_BOOSTS = {
    "interior defense": {"REB": 1.35, "BLK": 1.5},
    "rim protection": {"BLK": 1.5},
    "shooting": {"FG3_PCT": 1.5},
    "three-point shooting": {"FG3_PCT": 1.5},
    "playmaking": {"AST": 1.5},
    "perimeter defense": {"STL": 1.5},
    "rebounding": {"REB": 1.5},
}


def reason_need_weights(
    parsed_query,
    need_df: pd.DataFrame,
    allowed_metrics,
    use_llm: bool = True,
) -> NeedReasoningResult:
    """Return validated tactical multipliers and an adjusted need dataframe."""

    allowed = _normalize_allowed_metrics(allowed_metrics, need_df)
    fallback = _fallback_reasoning(parsed_query, need_df, allowed)
    if not use_llm:
        return fallback

    sentinel = {"_fallback": True}
    response = call_llm_json(
        [
            {"role": "system", "content": NEED_REASONER_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "parsed_goal": _query_goal(parsed_query),
                        "parsed_query": _query_payload(parsed_query),
                        "tool_a_needs": _need_rows(need_df),
                        "allowed_metrics": allowed,
                        "required_output": {
                            "tactical_interpretation": "short grounded string",
                            "metric_multipliers": {"REB": 1.0},
                            "explanations": {"REB": "short reason"},
                        },
                    },
                    ensure_ascii=True,
                ),
            },
        ],
        fallback=sentinel,
    )
    if response == sentinel or not isinstance(response, dict):
        return fallback

    result = _validate_llm_response(response, fallback, need_df, allowed)
    if not result.metric_multipliers:
        return fallback
    return result


def _fallback_reasoning(
    parsed_query, need_df: pd.DataFrame, allowed: dict[str, str]
) -> NeedReasoningResult:
    goal = _query_goal(parsed_query).lower()
    multipliers = {metric: 1.0 for metric in allowed}
    boosted: list[str] = []

    for goal_text, boosts in FALLBACK_BOOSTS.items():
        if goal_text in goal:
            for metric, multiplier in boosts.items():
                if metric in multipliers:
                    multipliers[metric] = multiplier
                    boosted.append(metric)

    if boosted:
        labels = [allowed[metric] for metric in boosted]
        tactical = "Fallback need reasoning emphasized " + ", ".join(labels) + "."
    else:
        tactical = "Fallback need reasoning kept Tool A need weights unchanged."

    explanations = {
        metric: (
            f"{allowed[metric]} is emphasized for the stated goal."
            if metric in boosted
            else f"{allowed[metric]} keeps the Tool A baseline emphasis."
        )
        for metric in allowed
    }

    return NeedReasoningResult(
        tactical_interpretation=tactical,
        metric_multipliers=multipliers,
        explanations=explanations,
        adjusted_need_df=_adjust_need_df(need_df, multipliers),
        warnings=[],
        used_fallback=True,
    )


def _validate_llm_response(
    response: dict[str, Any],
    fallback: NeedReasoningResult,
    need_df: pd.DataFrame,
    allowed: dict[str, str],
) -> NeedReasoningResult:
    warnings: list[str] = []
    raw_multipliers = response.get("metric_multipliers")
    if not isinstance(raw_multipliers, dict):
        return fallback

    multipliers = {metric: 1.0 for metric in allowed}
    for metric, raw_value in raw_multipliers.items():
        if metric not in allowed:
            warnings.append(f"Ignored invalid need-reasoning metric: {metric}.")
            continue
        multipliers[metric] = _clamp_multiplier(raw_value)

    raw_explanations = response.get("explanations")
    explanations = dict(fallback.explanations)
    if isinstance(raw_explanations, dict):
        for metric, explanation in raw_explanations.items():
            if metric in allowed and isinstance(explanation, str) and explanation.strip():
                explanations[metric] = explanation.strip()
            elif metric not in allowed:
                warnings.append(f"Ignored explanation for invalid metric: {metric}.")

    tactical = response.get("tactical_interpretation")
    if not isinstance(tactical, str) or not tactical.strip():
        tactical = fallback.tactical_interpretation

    return NeedReasoningResult(
        tactical_interpretation=tactical.strip(),
        metric_multipliers=multipliers,
        explanations=explanations,
        adjusted_need_df=_adjust_need_df(need_df, multipliers),
        warnings=warnings,
        used_fallback=False,
    )


def _adjust_need_df(need_df: pd.DataFrame, multipliers: dict[str, float]) -> pd.DataFrame:
    adjusted = need_df.copy()
    adjusted["adjusted_need_weight"] = adjusted.apply(
        lambda row: row["need_weight"] * multipliers.get(row["metric"], 1.0),
        axis=1,
    )
    return adjusted


def _normalize_allowed_metrics(allowed_metrics, need_df: pd.DataFrame) -> dict[str, str]:
    need_labels = dict(zip(need_df["metric"], need_df.get("label", need_df["metric"])))
    if isinstance(allowed_metrics, dict):
        candidates = allowed_metrics
    else:
        candidates = {metric: need_labels.get(metric, metric) for metric in allowed_metrics}

    need_metric_set = set(need_df["metric"])
    return {
        metric: str(label)
        for metric, label in candidates.items()
        if metric in need_metric_set
    }


def _need_rows(need_df: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        column
        for column in ["metric", "label", "need_weight", "team_value", "league_mean", "z_score"]
        if column in need_df.columns
    ]
    return need_df[columns].to_dict(orient="records")


def _query_payload(parsed_query) -> dict[str, Any]:
    if isinstance(parsed_query, AnalysisRequest):
        return {
            "team": parsed_query.team_name,
            "goal": parsed_query.goal,
            "top_k": parsed_query.top_k,
            "recent_games": parsed_query.recent_games,
            "ranking_mode": parsed_query.ranking_mode,
            "unavailable_constraints": list(parsed_query.unavailable_constraints),
        }
    if isinstance(parsed_query, dict):
        return dict(parsed_query)
    return {}


def _query_goal(parsed_query) -> str:
    if isinstance(parsed_query, AnalysisRequest):
        return parsed_query.goal or ""
    if isinstance(parsed_query, dict):
        return str(parsed_query.get("goal") or "")
    return ""


def _clamp_multiplier(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 1.0
    return min(max(numeric, 0.5), 2.0)
