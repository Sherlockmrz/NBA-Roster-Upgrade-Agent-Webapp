"""Query-aware tool selection for Streamlit display control."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from nba_agent.agentic.tool_registry import (
    ALWAYS_VISIBLE_TOOLS,
    FINAL_SUMMARY,
    FULL_TOOL_PIPELINE,
    GROUNDED_QA,
    NEED_REASONING,
    SELECTABLE_TOOL_IDS,
    SENSITIVITY,
    TOOL_A,
    TOOL_B,
    TOOL_C,
    TOOL_REGISTRY,
    ZERO_SHOT_EVALUATION,
)
from nba_agent.llm.client import call_llm_json_result


@dataclass(frozen=True)
class ToolSelectionResult:
    """Validated tool selection used to decide which UI sections are visible."""

    selected_tool_ids: tuple[str, ...]
    model_selected_tool_ids: tuple[str, ...]
    required_dependency_ids: tuple[str, ...]
    skipped_tool_ids: tuple[str, ...]
    rationales: dict[str, str] = field(default_factory=dict)
    skipped_rationales: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    validation_note: str = "Python validator confirmed all selected tools are allowed and dependency-safe."
    used_fallback: bool = False
    source: str = "deterministic"
    model: str = ""


def select_tools_for_query(
    user_query: str,
    parsed_query: Any | None = None,
    use_llm: bool = True,
) -> ToolSelectionResult:
    """Return validated tool selections for the current query."""

    fallback = deterministic_tool_selection(user_query, parsed_query)
    if not use_llm:
        return fallback

    llm_payload = _call_llm_tool_selector(user_query, parsed_query, fallback)
    if not llm_payload.ok:
        return ToolSelectionResult(
            **{
                **fallback.__dict__,
                "used_fallback": True,
                "source": "deterministic",
                "model": llm_payload.model,
                "warnings": [
                    *fallback.warnings,
                    f"LLM tool selection unavailable; deterministic selection used. {llm_payload.error_type}".strip(),
                ],
            }
        )

    content = llm_payload.content
    if not isinstance(content, dict):
        return ToolSelectionResult(
            **{
                **fallback.__dict__,
                "used_fallback": True,
                "source": "deterministic",
                "model": llm_payload.model,
                "warnings": [*fallback.warnings, "LLM tool selection returned invalid JSON shape."],
            }
        )

    selected = content.get("selected_tools", [])
    if not isinstance(selected, (list, tuple)):
        return ToolSelectionResult(
            **{
                **fallback.__dict__,
                "used_fallback": True,
                "source": "deterministic",
                "model": llm_payload.model,
                "warnings": [*fallback.warnings, "LLM tool selection returned invalid selected_tools."],
            }
        )
    selected = _sanitize_selected_tools_for_query(selected, query=_normalize_text(user_query))
    rationales = content.get("rationales", {})
    skipped_rationales = content.get("skipped_tools", {})
    result = validate_tool_selection(
        selected,
        rationales if isinstance(rationales, dict) else {},
        skipped_rationales if isinstance(skipped_rationales, dict) else {},
        source="llm",
        used_fallback=False,
        model=llm_payload.model,
    )
    if not result.selected_tool_ids:
        return fallback
    return result


def deterministic_tool_selection(
    user_query: str,
    parsed_query: Any | None = None,
) -> ToolSelectionResult:
    """Select tools with transparent keyword rules."""

    query = _normalize_text(user_query)
    goal = _normalize_text(getattr(parsed_query, "goal", "") or "")
    selected: list[str] = []
    rationales: dict[str, str] = {}

    if _asks_for_zero_shot_evaluation(query):
        selected.append(ZERO_SHOT_EVALUATION)
        rationales[ZERO_SHOT_EVALUATION] = "The query asks to compare the tool pipeline against a zero-shot baseline."
    elif _asks_for_player_recommendations(query):
        selected.extend([TOOL_A, TOOL_B, TOOL_C])
        rationales[TOOL_A] = "The query asks for player recommendations, so team needs must be diagnosed first."
        rationales[TOOL_B] = "Player candidates need dataset-backed strength representations."
        rationales[TOOL_C] = "The query asks for ranked roster recommendations."
        if _should_include_need_reasoning(query, goal):
            selected.append(NEED_REASONING)
            rationales[NEED_REASONING] = "The query includes or implies a tactical roster goal, so need weights should be interpreted before ranking."
    elif _asks_for_team_need_reasoning(query, goal):
        selected.extend([TOOL_A, NEED_REASONING])
        rationales[TOOL_A] = "The query asks about team needs, so Tool A diagnoses the weakness profile."
        rationales[NEED_REASONING] = "The query asks how a basketball goal changes need emphasis."
    elif _asks_for_weakness_only(query):
        selected.append(TOOL_A)
        rationales[TOOL_A] = "The query asks to diagnose team weaknesses."
    else:
        selected.extend([TOOL_A, NEED_REASONING, TOOL_B, TOOL_C])
        rationales[TOOL_A] = "Default roster-upgrade questions begin with team need diagnosis."
        rationales[NEED_REASONING] = "Default roster-upgrade questions use goal-aware need interpretation."
        rationales[TOOL_B] = "Default roster-upgrade questions need player strength vectors."
        rationales[TOOL_C] = "Default roster-upgrade questions return ranked player fits."

    selected.append(FINAL_SUMMARY)
    rationales[FINAL_SUMMARY] = "A final summary is always shown after a run."

    if _asks_for_stability(query) and _asks_for_player_recommendations(query):
        selected.append(SENSITIVITY)
        rationales[SENSITIVITY] = "The query asks whether the ranking is robust or stable."

    if _asks_for_qa(query):
        selected.append(GROUNDED_QA)
        rationales[GROUNDED_QA] = "The query asks to keep a grounded Q&A section for follow-up questions."

    return validate_tool_selection(
        selected,
        rationales,
        {},
        source="deterministic",
        used_fallback=True,
    )


def validate_tool_selection(
    selected_tool_ids: list[str] | tuple[str, ...],
    rationales: dict[str, str] | None = None,
    skipped_rationales: dict[str, str] | None = None,
    source: str = "deterministic",
    used_fallback: bool = False,
    model: str = "",
) -> ToolSelectionResult:
    """Remove unknown tools and add required dependencies in pipeline order."""

    rationale_map = dict(rationales or {})
    skipped_map = dict(skipped_rationales or {})
    warnings: list[str] = []
    model_selected: list[str] = []

    for tool_id in selected_tool_ids:
        normalized = str(tool_id).strip()
        if normalized not in SELECTABLE_TOOL_IDS:
            if normalized not in ALWAYS_VISIBLE_TOOLS:
                warnings.append(f"Unknown tool '{normalized}' was ignored.")
            continue
        if normalized not in model_selected:
            model_selected.append(normalized)

    if FINAL_SUMMARY not in model_selected:
        model_selected.append(FINAL_SUMMARY)
        rationale_map.setdefault(FINAL_SUMMARY, "A final summary is always shown after a run.")

    expanded = set(model_selected)
    dependency_ids: set[str] = set()
    changed = True
    while changed:
        changed = False
        for tool_id in list(expanded):
            for dependency in TOOL_REGISTRY[tool_id].dependencies:
                if dependency in ALWAYS_VISIBLE_TOOLS:
                    continue
                if dependency not in expanded:
                    expanded.add(dependency)
                    dependency_ids.add(dependency)
                    changed = True

    ordered_selected = tuple(
        definition.tool_id
        for definition in FULL_TOOL_PIPELINE
        if definition.tool_id in expanded and definition.tool_id not in ALWAYS_VISIBLE_TOOLS
    )
    ordered_model_selected = tuple(
        definition.tool_id
        for definition in FULL_TOOL_PIPELINE
        if definition.tool_id in model_selected and definition.tool_id not in ALWAYS_VISIBLE_TOOLS
    )
    ordered_dependencies = tuple(
        definition.tool_id
        for definition in FULL_TOOL_PIPELINE
        if definition.tool_id in dependency_ids and definition.tool_id not in model_selected
    )
    skipped = tuple(
        definition.tool_id
        for definition in FULL_TOOL_PIPELINE
        if definition.tool_id not in ordered_selected and definition.tool_id not in ALWAYS_VISIBLE_TOOLS
    )

    for dependency in ordered_dependencies:
        rationale_map.setdefault(
            dependency,
            f"Required dependency for {TOOL_REGISTRY[dependency].name}.",
        )
    for tool_id in skipped:
        skipped_map.setdefault(tool_id, "The query did not explicitly require this output.")

    return ToolSelectionResult(
        selected_tool_ids=ordered_selected,
        model_selected_tool_ids=ordered_model_selected,
        required_dependency_ids=ordered_dependencies,
        skipped_tool_ids=skipped,
        rationales=rationale_map,
        skipped_rationales=skipped_map,
        warnings=warnings,
        used_fallback=used_fallback,
        source=source,
        model=model,
    )


def _call_llm_tool_selector(user_query: str, parsed_query: Any, fallback: ToolSelectionResult):
    allowed = [
        {
            "id": definition.tool_id,
            "name": definition.name,
            "description": definition.description,
            "dependencies": list(definition.dependencies),
        }
        for definition in FULL_TOOL_PIPELINE
        if definition.tool_id not in ALWAYS_VISIBLE_TOOLS
    ]
    fallback_payload = {
        "selected_tools": list(fallback.model_selected_tool_ids),
        "rationales": fallback.rationales,
        "skipped_tools": fallback.skipped_rationales,
    }
    return call_llm_json_result(
        [
            {
                "role": "system",
                "content": (
                    "Return strict JSON only. Select display tools for an NBA roster agent. "
                    "Use only allowed tool ids. Do not invent tools, stats, salary, injuries, contracts, news, or rumors. "
                    "Always select final_scouting_summary. Only select Tool B and Tool C if the user explicitly asks "
                    "for player recommendations, player ranking, player comparison, best fits, target players, or top players. "
                    "Do not select Tool B or Tool C for a query that only asks about team needs, weaknesses, or which metrics matter. "
                    "For zero-shot baseline comparison requests, select zero_shot_evaluation and final_scouting_summary. "
                    "If unsure, omit the tool. Python will validate dependencies."
                ),
            },
            {
                "role": "user",
                "content": (
                    "User query:\n"
                    f"{user_query}\n\n"
                    f"Parsed query:\n{_parsed_query_payload(parsed_query)}\n\n"
                    f"Allowed tools:\n{allowed}\n\n"
                    "Return JSON with keys selected_tools, rationales, skipped_tools. "
                    "selected_tools must be a list of allowed ids. rationales and skipped_tools "
                    "must be objects keyed by tool id."
                ),
            },
        ],
        fallback=fallback_payload,
    )


def _parsed_query_payload(parsed_query: Any) -> dict[str, Any]:
    if parsed_query is None:
        return {}
    return {
        "team": getattr(parsed_query, "team_name", None),
        "goal": getattr(parsed_query, "goal", None),
        "top_k": getattr(parsed_query, "top_k", None),
        "recent_games": getattr(parsed_query, "recent_games", None),
        "min_games": getattr(parsed_query, "min_games", None),
        "min_avg_minutes": getattr(parsed_query, "min_avg_minutes", None),
        "exclude_current_team": getattr(parsed_query, "exclude_current_team", None),
    }


def _normalize_text(text: str) -> str:
    normalized = str(text or "").lower().replace("-", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def _asks_for_player_recommendations(query: str) -> bool:
    patterns = (
        r"\brecommend\b.*\b(players?|targets?|fits?)\b",
        r"\brank\s+players\b",
        r"\brank\s+candidate\s+players\b",
        r"\branked\s+players\b",
        r"\bcandidate\s+players\b.*\bfit\s+score\b",
        r"\btop\s+\d+\s+players\b",
        r"\btop\s+players\b",
        r"\bbest\s+fits\b",
        r"\bbest\s+fit\b",
        r"\btarget\s+players\b",
        r"\bwhich\s+players\b",
        r"\bfind\s+players\b",
        r"\bplayers\s+should\b",
    )
    return any(re.search(pattern, query) for pattern in patterns)


def _asks_for_zero_shot_evaluation(query: str) -> bool:
    return (
        "zero-shot" in query
        or "zero shot" in query
        or ("baseline" in query and "compare" in query)
        or ("tool pipeline" in query and "compare" in query)
    )


def _should_include_need_reasoning(query: str, goal: str) -> bool:
    if _asks_for_direct_fit_score_ranking(query):
        return False
    return True


def _asks_for_direct_fit_score_ranking(query: str) -> bool:
    return "fit score" in query and ("rank candidate players" in query or "candidate players" in query)


def _sanitize_selected_tools_for_query(
    selected_tool_ids: list[str] | tuple[str, ...],
    query: str,
) -> list[str]:
    selected = [str(tool_id).strip() for tool_id in selected_tool_ids]
    if _asks_for_zero_shot_evaluation(query):
        return [tool_id for tool_id in selected if tool_id in {ZERO_SHOT_EVALUATION, FINAL_SUMMARY}]
    if _asks_for_player_recommendations(query):
        if _asks_for_direct_fit_score_ranking(query):
            return [tool_id for tool_id in selected if tool_id != NEED_REASONING]
        return selected
    player_only_tools = {TOOL_B, TOOL_C, SENSITIVITY}
    return [tool_id for tool_id in selected if tool_id not in player_only_tools]


def _asks_for_weakness_only(query: str) -> bool:
    weakness_terms = ("weakness", "weaknesses", "team needs", "diagnose", "diagnosis", "lacking", "lack")
    return any(term in query for term in weakness_terms) and not _asks_for_player_recommendations(query)


def _asks_for_team_need_reasoning(query: str, goal: str) -> bool:
    reasoning_terms = (
        "explain",
        "matter",
        "emphasize",
        "need weights",
        "which needs",
        "needs matter",
        "team needs",
        "metrics matter",
        "goal is",
    )
    goal_terms = (
        "interior defense",
        "shooting",
        "playmaking",
        "rebounding",
        "perimeter defense",
        goal,
    )
    return any(term in query for term in reasoning_terms) and any(term and term in query for term in goal_terms)


def _asks_for_stability(query: str) -> bool:
    return any(term in query for term in ("robust", "stable", "stability", "sensitivity", "perturb"))


def _asks_for_qa(query: str) -> bool:
    return any(term in query for term in ("q&a", "qa", "follow-up", "follow up", "chat", "question"))
