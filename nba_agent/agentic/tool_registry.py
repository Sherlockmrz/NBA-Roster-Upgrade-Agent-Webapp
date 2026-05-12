"""Display registry for the app's agent pipeline tools."""

from __future__ import annotations

from dataclasses import dataclass


USER_QUERY = "user_query"
PARSED_QUERY = "parsed_query"
AGENTIC_TOOL_SELECTION = "agentic_tool_selection"
TOOL_A = "tool_a_need_diagnosis"
NEED_REASONING = "llm_need_reasoning"
TOOL_B = "tool_b_player_strength"
TOOL_C = "tool_c_fit_ranking"
SENSITIVITY = "sensitivity_check"
FINAL_SUMMARY = "final_scouting_summary"
GROUNDED_QA = "grounded_qa"
ZERO_SHOT_EVALUATION = "zero_shot_evaluation"

ALWAYS_VISIBLE_TOOLS = (USER_QUERY, PARSED_QUERY, AGENTIC_TOOL_SELECTION)


@dataclass(frozen=True)
class ToolDefinition:
    """UI-facing description of one pipeline node."""

    tool_id: str
    name: str
    description: str
    dependencies: tuple[str, ...] = ()


FULL_TOOL_PIPELINE: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        USER_QUERY,
        "User Query",
        "The natural-language roster question supplied by the user.",
    ),
    ToolDefinition(
        PARSED_QUERY,
        "Parsed Query",
        "Validated team, goal, and filter constraints extracted from the query.",
        dependencies=(USER_QUERY,),
    ),
    ToolDefinition(
        AGENTIC_TOOL_SELECTION,
        "Agentic Tool Selection",
        "The planner decides which result sections should be visible for this query.",
        dependencies=(PARSED_QUERY,),
    ),
    ToolDefinition(
        TOOL_A,
        "Tool A – Team Need Diagnosis",
        "Diagnoses recent team weaknesses and converts them into need weights.",
        dependencies=(PARSED_QUERY,),
    ),
    ToolDefinition(
        NEED_REASONING,
        "LLM Need Reasoning",
        "Interprets the basketball goal and adjusts need weights with bounded multipliers.",
        dependencies=(TOOL_A,),
    ),
    ToolDefinition(
        TOOL_B,
        "Tool B – Player Strength Representation",
        "Builds candidate player strength vectors from the loaded dataset.",
        dependencies=(TOOL_A,),
    ),
    ToolDefinition(
        TOOL_C,
        "Tool C – Fit Ranking",
        "Ranks player candidates by matching need weights to player strengths.",
        dependencies=(TOOL_B,),
    ),
    ToolDefinition(
        SENSITIVITY,
        "Sensitivity Check",
        "Tests whether recommendations stay stable under small need-weight changes.",
        dependencies=(TOOL_C,),
    ),
    ToolDefinition(
        FINAL_SUMMARY,
        "Final Scouting Summary",
        "Summarizes the computed outputs in concise scouting language.",
        dependencies=(PARSED_QUERY,),
    ),
    ToolDefinition(
        GROUNDED_QA,
        "Grounded Q&A",
        "Answers follow-up questions using only the current AgentResult.",
        dependencies=(PARSED_QUERY,),
    ),
    ToolDefinition(
        ZERO_SHOT_EVALUATION,
        "Zero-shot Baseline Comparison / Evaluation",
        "Compares the tool pipeline against a zero-shot LLM baseline in the Evaluation tab.",
    ),
)

TOOL_REGISTRY = {definition.tool_id: definition for definition in FULL_TOOL_PIPELINE}

SELECTABLE_TOOL_IDS = tuple(
    definition.tool_id
    for definition in FULL_TOOL_PIPELINE
    if definition.tool_id not in ALWAYS_VISIBLE_TOOLS
)
