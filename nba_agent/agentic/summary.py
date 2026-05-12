"""Display-safe summaries for selected agentic tool views."""

from __future__ import annotations

import pandas as pd

from nba_agent.agentic.tool_registry import NEED_REASONING, TOOL_C
from nba_agent.schemas import AgentResult, ScoutingSummaryResult


def summary_for_selected_tools(
    agent_result: AgentResult,
    selected_tool_ids: tuple[str, ...] | list[str] | set[str],
) -> ScoutingSummaryResult:
    """Return a summary scoped to the selected visible tools."""

    selected = set(selected_tool_ids)
    if TOOL_C in selected:
        return agent_result.scouting_summary
    return _need_only_summary(agent_result, include_need_reasoning=NEED_REASONING in selected)


def _need_only_summary(
    agent_result: AgentResult,
    include_need_reasoning: bool,
) -> ScoutingSummaryResult:
    need_df = (
        agent_result.need_reasoning.adjusted_need_df
        if include_need_reasoning and not agent_result.need_reasoning.adjusted_need_df.empty
        else agent_result.need_df
    )
    top_needs = _top_need_labels(need_df)
    needs_text = ", ".join(top_needs) if top_needs else "the available team-need signals"
    goal = agent_result.parsed_query.goal or "the requested basketball goal"
    team = agent_result.parsed_query.team_name

    if include_need_reasoning:
        reasoning_text = agent_result.need_reasoning.tactical_interpretation
        executive = (
            f"{team} was evaluated for {goal}. The selected view focuses on team diagnosis "
            f"and need reasoning: the most relevant needs are {needs_text}. {reasoning_text} "
            "This summary stays focused on team needs because candidate scoring was not selected."
        )
        takeaways = [
            f"Tool A diagnosis points most strongly to {needs_text}.",
            f"Need reasoning interprets the tactical goal as: {reasoning_text}",
            "Scope: this selected view stops at team-need analysis and does not discuss candidate scoring.",
        ]
    else:
        executive = (
            f"{team} was evaluated for {goal}. The selected view focuses on Tool A team "
            f"diagnosis, where the most relevant needs are {needs_text}."
        )
        takeaways = [
            f"Tool A diagnosis points most strongly to {needs_text}.",
            "Need reasoning was not selected for this query.",
            "Scope: this selected view stops at team-need analysis.",
        ]

    limitations = (
        "This display summary is scoped to the tools selected for the current query."
    )
    return ScoutingSummaryResult(
        executive_summary=executive,
        key_takeaways=takeaways,
        limitations_note=limitations,
        used_fallback=True,
        warnings=[],
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
