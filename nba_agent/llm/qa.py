"""Grounded Q&A over the current agent result."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

import pandas as pd

from nba_agent.llm.client import call_llm_text, is_llm_available
from nba_agent.llm.prompts import GROUNDED_QA_PROMPT
from nba_agent.schemas import AgentResult


SALARY_UNAVAILABLE = "Salary data is unavailable in the current dataset."

_SALARY_TERMS = ("salary", "salaries", "cap", "payroll")
_UNAVAILABLE_TERMS = (
    "contract",
    "contracts",
    "injury",
    "injuries",
    "trade rumor",
    "trade rumors",
    "rumor",
    "rumors",
    "current news",
    "nba news",
    "advanced stat",
    "advanced stats",
)


def answer_grounded_question(
    agent_result: AgentResult,
    question: str,
    use_llm: bool = True,
) -> str:
    """Answer a question using only the provided AgentResult."""

    clean_question = question.strip()
    if not clean_question:
        return "Ask a question about the current pipeline outputs."
    if _asks_about_salary(clean_question):
        return SALARY_UNAVAILABLE

    fallback = _deterministic_answer(agent_result, clean_question)
    if not use_llm or not is_llm_available():
        return fallback

    context = build_qa_context(agent_result)
    answer = call_llm_text(
        [
            {"role": "system", "content": GROUNDED_QA_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"question": clean_question, "current_agent_result": context},
                    ensure_ascii=True,
                ),
            },
        ],
        fallback=fallback,
    )
    if not isinstance(answer, str) or not answer.strip():
        return fallback
    return answer.strip()


def build_qa_context(agent_result: AgentResult) -> dict[str, Any]:
    """Return a compact, JSON-friendly context scoped to the current run."""

    adjusted_need_df = agent_result.need_reasoning.adjusted_need_df
    ranked_records = _records(
        agent_result.ranked_df,
        [
            "PLAYER_NAME",
            "CURRENT_TEAM",
            "fit_score",
            "best_match",
            "GP",
            "AVG_MIN",
        ],
        rows=10,
    )
    return {
        "parsed_query": asdict(agent_result.parsed_query),
        "agent_plan": agent_result.agent_plan,
        "need_df": _records(agent_result.need_df, ["metric", "label", "need_weight"]),
        "adjusted_need_df": _records(
            adjusted_need_df,
            ["metric", "label", "need_weight", "adjusted_need_weight"],
        ),
        "need_reasoning": {
            "tactical_interpretation": agent_result.need_reasoning.tactical_interpretation,
            "metric_multipliers": agent_result.need_reasoning.metric_multipliers,
            "explanations": agent_result.need_reasoning.explanations,
            "used_fallback": agent_result.need_reasoning.used_fallback,
        },
        "ranked_df": ranked_records,
        "final_ranked_df": ranked_records,
        "feasibility_output": "Feasibility critique is not implemented in this version.",
        "sensitivity_output": {
            "stability_label": agent_result.sensitivity.stability_label,
            "top_k_overlap": agent_result.sensitivity.top_k_overlap,
            "original_top_players": agent_result.sensitivity.original_top_players,
            "perturbed_top_players": agent_result.sensitivity.perturbed_top_players,
            "explanation": agent_result.sensitivity.explanation,
        },
        "final_summary": {
            "executive_summary": agent_result.scouting_summary.executive_summary,
            "key_takeaways": agent_result.scouting_summary.key_takeaways,
            "limitations_note": agent_result.scouting_summary.limitations_note,
        },
    }


def _deterministic_answer(agent_result: AgentResult, question: str) -> str:
    normalized = question.lower()
    if "first" in normalized or "ranked first" in normalized or "ranked #1" in normalized:
        return _first_player_answer(agent_result)
    if "rebounding" in normalized and ("need" in normalized or "lack" in normalized):
        return _need_answer(agent_result, "REB", "rebounding")
    if "need reasoning" in normalized or "llm need" in normalized or "what changed after llm" in normalized:
        return _need_reasoning_answer(agent_result)
    if "feasibility" in normalized:
        return (
            "Feasibility critique is not implemented in this version. The current "
            "answer is grounded in Tool A needs, Tool B strengths, Tool C fit "
            "ranking, need reasoning, sensitivity, and the final summary."
        )
    if "hidden gem" in normalized or "hidden gems" in normalized:
        top_player = _top_player_name(agent_result)
        if top_player:
            return (
                "Hidden-gem labels are not implemented in this version. The current "
                f"computed fit ranking has {top_player} first, but that is not a "
                "hidden-gem classification."
            )
        return "Hidden-gem labels are not implemented in this version."
    if "stable" in normalized or "stability" in normalized or "robust" in normalized:
        return _stability_answer(agent_result)
    if any(term in normalized for term in _UNAVAILABLE_TERMS):
        return (
            "Contracts, injuries, trade rumors, current NBA news, and unavailable "
            "advanced stats are not available in the current dataset."
        )
    if "missing" in normalized or "unavailable" in normalized or "data" in normalized:
        return (
            "The current dataset does not include salary data, contracts, injuries, "
            "trade rumors, current NBA news, or unavailable advanced stats. Answers "
            "are limited to the computed pipeline outputs shown in this app."
        )
    return _general_grounded_answer(agent_result)


def _first_player_answer(agent_result: AgentResult) -> str:
    if agent_result.ranked_df.empty or "PLAYER_NAME" not in agent_result.ranked_df.columns:
        return "No ranked player is available for this run."
    row = agent_result.ranked_df.iloc[0]
    player = str(row.get("PLAYER_NAME", "the first player"))
    fit_score = row.get("fit_score")
    best_match = str(row.get("best_match", "the weighted team needs"))
    team = str(row.get("CURRENT_TEAM", "unknown team"))
    score_text = _format_number(fit_score)
    if score_text:
        return (
            f"{player} is ranked first because Tool C gave him the highest computed "
            f"fit score ({score_text}) against the current adjusted needs. His listed "
            f"match is {best_match}, and his current team in the dataset is {team}."
        )
    return (
        f"{player} is ranked first because Tool C placed him at the top of the "
        f"computed fit ranking. His listed match is {best_match}."
    )


def _need_answer(agent_result: AgentResult, metric: str, label: str) -> str:
    row = _metric_row(agent_result.need_df, metric)
    adjusted_row = _metric_row(agent_result.need_reasoning.adjusted_need_df, metric)
    if row is None:
        return f"{label.title()} is not present as a computed Tool A need for this run."
    base_weight = _format_number(row.get("need_weight"))
    adjusted_weight = _format_number(
        adjusted_row.get("adjusted_need_weight") if adjusted_row is not None else None
    )
    if adjusted_weight:
        return (
            f"Tool A includes {label} as a computed need with base weight {base_weight}. "
            f"After need reasoning, its adjusted need weight is {adjusted_weight}."
        )
    return f"Tool A includes {label} as a computed need with weight {base_weight}."


def _need_reasoning_answer(agent_result: AgentResult) -> str:
    reasoning = agent_result.need_reasoning
    changed = [
        f"{metric} x{multiplier:.2f}"
        for metric, multiplier in reasoning.metric_multipliers.items()
        if abs(float(multiplier) - 1.0) > 0.001
    ]
    changed_text = ", ".join(changed) if changed else "no metric multipliers changed from 1.00"
    return (
        f"Need Reasoning interpreted the goal as: {reasoning.tactical_interpretation} "
        f"Validated multiplier changes: {changed_text}."
    )


def _stability_answer(agent_result: AgentResult) -> str:
    sensitivity = agent_result.sensitivity
    return (
        f"The recommendation is {sensitivity.stability_label} with "
        f"{sensitivity.top_k_overlap:.0%} top-k overlap after small need-weight "
        f"perturbations. {sensitivity.explanation}"
    )


def _general_grounded_answer(agent_result: AgentResult) -> str:
    top_player = _top_player_name(agent_result) or "no ranked player"
    needs = _top_need_labels(agent_result.need_reasoning.adjusted_need_df)
    needs_text = ", ".join(needs) if needs else "no clear computed needs"
    return (
        f"For this run, the parsed team is {agent_result.parsed_query.team_name}, "
        f"the main computed needs are {needs_text}, and the first ranked player is "
        f"{top_player}. Ask about the first player, need reasoning, stability, or "
        "missing data for a more specific grounded answer."
    )


def _top_player_name(agent_result: AgentResult) -> str:
    if agent_result.ranked_df.empty or "PLAYER_NAME" not in agent_result.ranked_df.columns:
        return ""
    return str(agent_result.ranked_df.iloc[0].get("PLAYER_NAME", "")).strip()


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
    return need_df.sort_values(weight_column, ascending=False)["label"].head(3).astype(str).tolist()


def _metric_row(df: pd.DataFrame, metric: str) -> dict[str, Any] | None:
    if df.empty or "metric" not in df.columns:
        return None
    match = df[df["metric"].astype(str).str.upper() == metric.upper()]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _records(df: pd.DataFrame, columns: list[str], rows: int = 20) -> list[dict[str, Any]]:
    if df.empty:
        return []
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return []
    return df[existing].head(rows).to_dict(orient="records")


def _format_number(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def _asks_about_salary(text: str) -> bool:
    normalized = text.lower()
    return any(term in normalized for term in _SALARY_TERMS)
