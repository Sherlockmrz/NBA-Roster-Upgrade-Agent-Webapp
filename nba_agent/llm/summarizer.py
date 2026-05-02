"""Final grounded scouting summary generation."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from nba_agent.llm.client import call_llm_json
from nba_agent.llm.prompts import SCOUTING_SUMMARIZER_PROMPT
from nba_agent.schemas import (
    AnalysisRequest,
    NeedReasoningResult,
    ScoutingSummaryResult,
    SensitivityResult,
)


def summarize_scouting_report(
    parsed_query: AnalysisRequest,
    agent_plan: list[str],
    need_df: pd.DataFrame,
    adjusted_need_df: pd.DataFrame,
    need_reasoning: NeedReasoningResult,
    ranked_df: pd.DataFrame,
    sensitivity_output: SensitivityResult,
    use_llm: bool = True,
) -> ScoutingSummaryResult:
    """Return a validated final summary from computed pipeline outputs only."""

    fallback = _fallback_summary(
        parsed_query,
        need_df,
        adjusted_need_df,
        need_reasoning,
        ranked_df,
        sensitivity_output,
    )
    if not use_llm:
        return fallback

    sentinel = {"_fallback": True}
    response = call_llm_json(
        [
            {"role": "system", "content": SCOUTING_SUMMARIZER_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "parsed_query": _parsed_query_payload(parsed_query),
                        "agent_plan": agent_plan,
                        "need_df": _records(
                            need_df,
                            ["metric", "label", "need_weight"],
                        ),
                        "adjusted_need_df": _records(
                            adjusted_need_df,
                            ["metric", "label", "need_weight", "adjusted_need_weight"],
                        ),
                        "need_reasoning": {
                            "tactical_interpretation": need_reasoning.tactical_interpretation,
                            "metric_multipliers": need_reasoning.metric_multipliers,
                            "explanations": need_reasoning.explanations,
                            "used_fallback": need_reasoning.used_fallback,
                        },
                        "ranked_df": _records(
                            ranked_df,
                            ["PLAYER_NAME", "CURRENT_TEAM", "fit_score", "best_match"],
                        ),
                        "sensitivity_output": {
                            "stability_label": sensitivity_output.stability_label,
                            "top_k_overlap": sensitivity_output.top_k_overlap,
                            "original_top_players": sensitivity_output.original_top_players,
                            "perturbed_top_players": sensitivity_output.perturbed_top_players,
                            "explanation": sensitivity_output.explanation,
                        },
                        "limitations": fallback.limitations_note,
                    },
                    ensure_ascii=True,
                ),
            },
        ],
        fallback=sentinel,
    )
    if response == sentinel or not isinstance(response, dict):
        return fallback

    return _validate_summary_response(response, fallback)


def _fallback_summary(
    parsed_query: AnalysisRequest,
    need_df: pd.DataFrame,
    adjusted_need_df: pd.DataFrame,
    need_reasoning: NeedReasoningResult,
    ranked_df: pd.DataFrame,
    sensitivity_output: SensitivityResult,
) -> ScoutingSummaryResult:
    top_needs = _top_need_labels(adjusted_need_df if not adjusted_need_df.empty else need_df)
    top_players = ranked_df["PLAYER_NAME"].head(parsed_query.top_k).astype(str).tolist() if "PLAYER_NAME" in ranked_df.columns else []
    top_players_text = ", ".join(top_players) if top_players else "no eligible players"
    needs_text = ", ".join(top_needs) if top_needs else "no clear needs"
    goal = parsed_query.goal or "the requested roster goal"

    executive = (
        f"{parsed_query.team_name} was evaluated for {goal}. The main computed needs are "
        f"{needs_text}, and the top statistical fits are {top_players_text}."
    )
    takeaways = [
        f"Team needs: Tool A and adjusted weights point most strongly to {needs_text}.",
        f"Best fits: {top_players_text} rank highest because their strength profiles match the weighted needs.",
        (
            "Robustness: "
            f"{sensitivity_output.stability_label} with {sensitivity_output.top_k_overlap:.0%} top-k overlap after weight perturbation."
        ),
    ]
    limitations = (
        "This summary uses only computed dataset outputs. Salary data, contracts, injuries, "
        "trade rumors, and current NBA news are unavailable in the current dataset."
    )

    return ScoutingSummaryResult(
        executive_summary=executive,
        key_takeaways=takeaways,
        limitations_note=limitations,
        used_fallback=True,
        warnings=[],
    )


def _validate_summary_response(
    response: dict[str, Any], fallback: ScoutingSummaryResult
) -> ScoutingSummaryResult:
    warnings: list[str] = []
    executive = response.get("executive_summary")
    if not isinstance(executive, str) or not executive.strip():
        executive = fallback.executive_summary
        warnings.append("LLM summary omitted executive_summary; fallback text was used.")

    raw_takeaways = response.get("key_takeaways")
    if not isinstance(raw_takeaways, list):
        takeaways = fallback.key_takeaways
        warnings.append("LLM summary omitted key_takeaways; fallback takeaways were used.")
    else:
        takeaways = [str(item).strip() for item in raw_takeaways if str(item).strip()]
        if len(takeaways) < 3:
            takeaways = (takeaways + fallback.key_takeaways)[:3]
            warnings.append("LLM summary returned fewer than three takeaways; fallback takeaways filled the gap.")
        else:
            takeaways = takeaways[:3]

    limitations = response.get("limitations_note")
    if not isinstance(limitations, str) or not limitations.strip():
        limitations = fallback.limitations_note
        warnings.append("LLM summary omitted limitations_note; fallback limitations were used.")

    return ScoutingSummaryResult(
        executive_summary=executive.strip(),
        key_takeaways=takeaways,
        limitations_note=limitations.strip(),
        used_fallback=False,
        warnings=warnings,
    )


def _top_need_labels(need_df: pd.DataFrame) -> list[str]:
    if need_df.empty:
        return []
    weight_column = (
        "adjusted_need_weight"
        if "adjusted_need_weight" in need_df.columns
        else "need_weight"
    )
    if not {"label", weight_column}.issubset(need_df.columns):
        return []
    return (
        need_df.sort_values(weight_column, ascending=False)["label"]
        .head(3)
        .astype(str)
        .tolist()
    )


def _records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return []
    return df[existing].to_dict(orient="records")


def _parsed_query_payload(parsed_query: AnalysisRequest) -> dict[str, Any]:
    return {
        "team_name": parsed_query.team_name,
        "goal": parsed_query.goal,
        "top_k": parsed_query.top_k,
        "recent_games": parsed_query.recent_games,
        "min_games": parsed_query.min_games,
        "min_avg_minutes": parsed_query.min_avg_minutes,
        "exclude_current_team": parsed_query.exclude_current_team,
        "ranking_mode": parsed_query.ranking_mode,
        "unavailable_constraints": list(parsed_query.unavailable_constraints),
    }
