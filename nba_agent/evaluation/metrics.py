"""Dataset-grounded evaluation metrics for recommendation lists."""

from __future__ import annotations

from dataclasses import dataclass
import re
import string
from typing import Any

import pandas as pd

from nba_agent.schemas import AnalysisRequest, SensitivityResult
from nba_agent.tools.need_diagnosis import CORE_METRIC_LABELS, CORE_METRICS


@dataclass(frozen=True)
class RecommendationMetrics:
    """Aggregate metrics for one recommendation source."""

    found_rate: float
    constraint_satisfaction_rate: float
    average_fit_score_matched: float | None
    average_fit_score_penalized: float
    need_alignment_score: float | None
    explainability_score: float


_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


def normalize_player_name(name: str) -> str:
    """Normalize player names for conservative matching."""

    normalized = str(name or "").lower().translate(_PUNCT_TRANSLATION)
    return re.sub(r"\s+", " ", normalized).strip()


def build_scored_candidate_table(
    need_df: pd.DataFrame,
    player_strength_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score the current candidate table using the same Tool C dot product."""

    if player_strength_df.empty:
        return player_strength_df.copy()

    scored = player_strength_df.copy()
    weight_column = (
        "adjusted_need_weight"
        if "adjusted_need_weight" in need_df.columns
        else "need_weight"
    )
    need_map = dict(zip(need_df["metric"], need_df[weight_column]))
    scored["fit_score"] = 0.0
    for metric in CORE_METRICS:
        strength_column = f"{metric}_strength"
        if strength_column not in scored.columns:
            scored[strength_column] = 0.0
        scored["fit_score"] += need_map.get(metric, 0.0) * scored[strength_column]

    def best_match(row: pd.Series) -> str:
        parts = {}
        for metric in CORE_METRICS:
            parts[CORE_METRIC_LABELS[metric]] = (
                need_map.get(metric, 0.0) * row.get(f"{metric}_strength", 0.0)
            )
        best_two = sorted(parts.items(), key=lambda item: item[1], reverse=True)[:2]
        labels = [label for label, score in best_two if score > 0]
        return " + ".join(labels) if labels else "General fit"

    scored["best_match"] = scored.apply(best_match, axis=1)
    return scored


def match_recommended_players(
    recommendations: list[dict[str, Any]],
    candidate_table: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Match recommendation names to the current candidate table."""

    if candidate_table.empty or "PLAYER_NAME" not in candidate_table.columns:
        return [
            {
                "rank": item.get("rank", index + 1),
                "player_name": item.get("player_name", ""),
                "reasoning": item.get("reasoning", ""),
                "found": False,
                "matched_row": None,
            }
            for index, item in enumerate(recommendations)
        ]

    table = candidate_table.copy()
    table["_normalized_name"] = table["PLAYER_NAME"].map(normalize_player_name)
    matches = []
    for index, item in enumerate(recommendations):
        raw_name = str(item.get("player_name", ""))
        normalized = normalize_player_name(raw_name)
        matched_row = _match_one_player(normalized, table, item.get("current_team"))
        matches.append(
            {
                "rank": item.get("rank", index + 1),
                "player_name": raw_name,
                "reasoning": item.get("reasoning", ""),
                "found": matched_row is not None,
                "matched_row": matched_row,
            }
        )
    return matches


def compute_candidate_found_rate(matches: list[dict[str, Any]], top_k: int) -> float:
    if top_k <= 0:
        return 0.0
    return sum(1 for item in matches[:top_k] if item["found"]) / top_k


def compute_constraint_satisfaction_rate(
    matches: list[dict[str, Any]],
    parsed_query: AnalysisRequest,
    top_k: int,
) -> float:
    if top_k <= 0:
        return 0.0
    passed = 0
    for item in matches[:top_k]:
        row = item.get("matched_row")
        if row is not None and _passes_constraints(row, parsed_query):
            passed += 1
    return passed / top_k


def compute_average_fit_score(
    matches: list[dict[str, Any]],
    top_k: int,
    penalize_unmatched: bool = False,
) -> float | None:
    scores: list[float] = []
    for item in matches[:top_k]:
        row = item.get("matched_row")
        if row is None:
            if penalize_unmatched:
                scores.append(0.0)
            continue
        scores.append(_float_or_zero(row.get("fit_score")))
    if not scores:
        return 0.0 if penalize_unmatched else None
    if penalize_unmatched and len(scores) < top_k:
        scores.extend([0.0] * (top_k - len(scores)))
    return sum(scores) / len(scores)


def compute_need_alignment_score(
    matches: list[dict[str, Any]],
    need_df: pd.DataFrame,
    top_k: int,
    top_n_needs: int = 3,
) -> float | None:
    strength_columns = _top_need_strength_columns(need_df, top_n_needs)
    if not strength_columns:
        return None

    player_scores: list[float] = []
    for item in matches[:top_k]:
        row = item.get("matched_row")
        if row is None:
            continue
        values = [
            _float_or_zero(row.get(column))
            for column in strength_columns
            if column in row.index
        ]
        if values:
            player_scores.append(sum(values) / len(values))
    if not player_scores:
        return None
    return sum(player_scores) / len(player_scores)


def compute_explainability_score(checklist: dict[str, bool]) -> float:
    if not checklist:
        return 0.0
    return sum(1 for value in checklist.values() if value) / len(checklist)


def evaluate_recommendations(
    recommendations: list[dict[str, Any]],
    candidate_table: pd.DataFrame,
    parsed_query: AnalysisRequest,
    need_df: pd.DataFrame,
    top_k: int,
    explainability_checklist: dict[str, bool],
) -> tuple[RecommendationMetrics, list[dict[str, Any]]]:
    matches = match_recommended_players(recommendations, candidate_table)
    return (
        RecommendationMetrics(
            found_rate=compute_candidate_found_rate(matches, top_k),
            constraint_satisfaction_rate=compute_constraint_satisfaction_rate(
                matches, parsed_query, top_k
            ),
            average_fit_score_matched=compute_average_fit_score(
                matches, top_k, penalize_unmatched=False
            ),
            average_fit_score_penalized=compute_average_fit_score(
                matches, top_k, penalize_unmatched=True
            )
            or 0.0,
            need_alignment_score=compute_need_alignment_score(matches, need_df, top_k),
            explainability_score=compute_explainability_score(explainability_checklist),
        ),
        matches,
    )


def build_comparison_table(
    zero_metrics: RecommendationMetrics,
    tool_metrics: RecommendationMetrics,
    sensitivity: SensitivityResult,
) -> pd.DataFrame:
    """Build the main metric comparison table."""

    rows = [
        _rate_row(
            "Candidate Found Rate",
            zero_metrics.found_rate,
            tool_metrics.found_rate,
            "Fair audit metric",
            "Fair audit check: measures whether recommendations can be evaluated in the current dataset.",
        ),
        _rate_row(
            "Constraint Satisfaction Rate",
            zero_metrics.constraint_satisfaction_rate,
            tool_metrics.constraint_satisfaction_rate,
            "Fair audit metric",
            "Fair audit check: both outputs are tested against the same user constraints.",
        ),
        _average_row(
            "Average Tool C Fit Score",
            zero_metrics.average_fit_score_matched,
            tool_metrics.average_fit_score_matched,
            "Internal objective metric",
            (
                "Measures alignment with our explicit Tool C objective. Since the tool pipeline optimizes "
                "this score, this is an internal objective comparison, not an independent ground-truth evaluation."
            ),
        ),
        _average_row(
            "Penalized Average Tool C Fit Score",
            zero_metrics.average_fit_score_penalized,
            tool_metrics.average_fit_score_penalized,
            "Internal objective metric",
            "Penalizes unmatched recommendations while still measuring alignment with our explicit Tool C objective.",
        ),
        _average_row(
            "Need Alignment Score",
            zero_metrics.need_alignment_score,
            tool_metrics.need_alignment_score,
            "Internal objective metric",
            (
                "Measures alignment with diagnosed team needs under our metric mapping. This is useful for "
                "auditing tool-objective alignment, but it depends on the chosen need/strength mapping."
            ),
        ),
        {
            "Metric Type": "Fair audit metric",
            "Metric": "Robustness Check Available",
            "Zero-shot LLM": "N/A",
            "Our Tool Pipeline": f"{sensitivity.stability_label} ({sensitivity.top_k_overlap:.0%} overlap)",
            "Improvement": "audit improvement: added verification",
            "Interpretation": "Only the tool pipeline can test stability under need-weight perturbation.",
        },
        _rate_row(
            "Evidence Coverage / Explainability Score",
            zero_metrics.explainability_score,
            tool_metrics.explainability_score,
            "Fair audit metric",
            "Fair audit check: measures how much verifiable intermediate evidence supports each recommendation.",
        ),
    ]
    return pd.DataFrame(rows)


def build_player_comparison_table(
    zero_matches: list[dict[str, Any]],
    tool_matches: list[dict[str, Any]],
    parsed_query: AnalysisRequest,
    top_k: int,
) -> pd.DataFrame:
    rows = []
    for index in range(top_k):
        zero = zero_matches[index] if index < len(zero_matches) else {}
        tool = tool_matches[index] if index < len(tool_matches) else {}
        zero_row = zero.get("matched_row")
        tool_row = tool.get("matched_row")
        zero_name = zero.get("player_name", "")
        tool_name = tool.get("player_name", "")
        rows.append(
            {
                "Rank": index + 1,
                "Zero-shot player": zero_name,
                "Zero-shot reasoning": zero.get("reasoning", ""),
                "Zero-shot found in dataset": bool(zero.get("found")),
                "Zero-shot passes constraints": bool(
                    zero_row is not None and _passes_constraints(zero_row, parsed_query)
                ),
                "Zero-shot Tool C fit_score": _display_number(
                    zero_row.get("fit_score") if zero_row is not None else None
                ),
                "Tool pipeline player": tool_name,
                "Tool pipeline fit_score": _display_number(
                    tool_row.get("fit_score") if tool_row is not None else None
                ),
                "Tool pipeline best_match": (
                    tool_row.get("best_match") if tool_row is not None else ""
                ),
                "Same player?": "yes"
                if normalize_player_name(zero_name) == normalize_player_name(tool_name)
                and zero_name
                else "no",
            }
        )
    return pd.DataFrame(rows)


def tool_recommendations_from_ranked_df(ranked_df: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    if ranked_df.empty:
        return []
    rows = []
    for index, (_, row) in enumerate(ranked_df.head(top_k).iterrows(), start=1):
        rows.append(
            {
                "rank": index,
                "player_name": row.get("PLAYER_NAME", ""),
                "current_team": row.get("CURRENT_TEAM", ""),
                "reasoning": row.get("best_match", ""),
            }
        )
    return rows


def zero_shot_explainability_checklist() -> dict[str, bool]:
    return {
        "dataset_backed_filter_verification": False,
        "computed_fit_score": False,
        "team_need_diagnosis": False,
        "player_strength_vector_evidence": False,
        "robustness_check": False,
    }


def tool_pipeline_explainability_checklist(result) -> dict[str, bool]:
    return {
        "dataset_backed_filter_verification": not result.player_strength_df.empty,
        "computed_fit_score": "fit_score" in result.ranked_df.columns,
        "team_need_diagnosis": not result.need_df.empty,
        "player_strength_vector_evidence": any(
            column.endswith("_strength") for column in result.player_strength_df.columns
        ),
        "robustness_check": result.sensitivity is not None,
    }


def _match_one_player(
    normalized: str,
    table: pd.DataFrame,
    current_team: Any = None,
) -> pd.Series | None:
    exact = table[table["_normalized_name"] == normalized]
    if current_team and "CURRENT_TEAM" in exact.columns:
        team_exact = exact[exact["CURRENT_TEAM"].astype(str) == str(current_team)]
        if len(team_exact) == 1:
            return team_exact.iloc[0]
    if len(exact) == 1:
        return exact.iloc[0]
    contains = table[
        table["_normalized_name"].map(
            lambda candidate: bool(normalized)
            and (normalized in candidate or candidate in normalized)
        )
    ]
    if len(contains) == 1:
        return contains.iloc[0]
    return None


def _passes_constraints(row: pd.Series, parsed_query: AnalysisRequest) -> bool:
    if _float_or_zero(row.get("GP")) < parsed_query.min_games:
        return False
    if _float_or_zero(row.get("AVG_MIN")) < parsed_query.min_avg_minutes:
        return False
    if parsed_query.exclude_current_team and row.get("CURRENT_TEAM") == parsed_query.team_name:
        return False
    return True


def _top_need_strength_columns(need_df: pd.DataFrame, top_n_needs: int) -> list[str]:
    if need_df.empty or "metric" not in need_df.columns:
        return []
    weight_column = (
        "adjusted_need_weight"
        if "adjusted_need_weight" in need_df.columns
        else "need_weight"
    )
    if weight_column not in need_df.columns:
        return []
    top_metrics = (
        need_df.assign(_weight=pd.to_numeric(need_df[weight_column], errors="coerce").fillna(0.0))
        .sort_values("_weight", ascending=False)["metric"]
        .head(top_n_needs)
        .astype(str)
        .tolist()
    )
    return [f"{metric}_strength" for metric in top_metrics]


def _rate_row(
    metric: str,
    zero: float,
    tool: float,
    metric_type: str,
    interpretation: str,
) -> dict[str, str]:
    return {
        "Metric Type": metric_type,
        "Metric": metric,
        "Zero-shot LLM": f"{zero:.0%}",
        "Our Tool Pipeline": f"{tool:.0%}",
        "Improvement": f"audit improvement: {(tool - zero) * 100:+.0f} pp",
        "Interpretation": interpretation,
    }


def _average_row(
    metric: str,
    zero: float | None,
    tool: float | None,
    metric_type: str,
    interpretation: str,
) -> dict[str, str]:
    if zero is None:
        zero_text = "N/A"
    else:
        zero_text = f"{zero:.2f}"
    tool_text = "N/A" if tool is None else f"{tool:.2f}"
    if zero is None or tool is None:
        improvement = "N/A"
    else:
        absolute = tool - zero
        if abs(zero) > 0:
            relative = absolute / abs(zero) * 100
            improvement = f"objective-alignment difference: {absolute:+.2f} ({relative:+.0f}%)"
        else:
            improvement = f"objective-alignment difference: {absolute:+.2f} (relative N/A)"
    return {
        "Metric Type": metric_type,
        "Metric": metric,
        "Zero-shot LLM": zero_text,
        "Our Tool Pipeline": tool_text,
        "Improvement": improvement,
        "Interpretation": interpretation,
    }


def _display_number(value: Any) -> float | str:
    if value is None:
        return "N/A"
    return round(_float_or_zero(value), 2)


def _float_or_zero(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
