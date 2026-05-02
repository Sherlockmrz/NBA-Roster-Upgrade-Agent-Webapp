"""Top-level deterministic orchestration for the roster-upgrade pipeline."""

from __future__ import annotations

import re
from typing import Any

from nba_agent.data.loader import load_raw_data
from nba_agent.data.preprocessing import find_team_id, prepare_data
from nba_agent.llm.client import get_llm_status
from nba_agent.llm.parser import (
    find_team_in_query,
    get_parser_warnings,
    normalize_team_value,
    parse_user_query,
)
from nba_agent.llm.planner import get_planner_warnings, plan_tools
from nba_agent.llm.reasoner import reason_need_weights
from nba_agent.llm.summarizer import summarize_scouting_report
from nba_agent.schemas import AgentResult, AnalysisRequest, PreparedNBAData, TraceStep
from nba_agent.tools.fit_ranking import rank_players_by_fit
from nba_agent.tools.need_diagnosis import CORE_METRIC_LABELS, diagnose_team_needs
from nba_agent.tools.player_strength import build_player_strengths
from nba_agent.tools.sensitivity import run_sensitivity_check


DEFAULT_REQUEST = AnalysisRequest(team_name="Warriors")
DEFAULT_FILTERS = {
    "team": DEFAULT_REQUEST.team_name,
    "goal": DEFAULT_REQUEST.goal,
    "top_k": DEFAULT_REQUEST.top_k,
    "recent_games": DEFAULT_REQUEST.recent_games,
    "min_games": DEFAULT_REQUEST.min_games,
    "min_avg_minutes": DEFAULT_REQUEST.min_avg_minutes,
    "exclude_current_team": DEFAULT_REQUEST.exclude_current_team,
    "ranking_mode": DEFAULT_REQUEST.ranking_mode,
}


GOAL_KEYWORDS = [
    "interior defense",
    "rim protection",
    "rebounding",
    "playmaking",
    "perimeter defense",
    "three-point shooting",
    "shooting",
]

UNAVAILABLE_CONSTRAINT_WARNINGS = {
    "salary": "Salary data is unavailable in the current dataset.",
    "contracts": "Contract data is unavailable in the current dataset.",
    "injuries": "Injury data is unavailable in the current dataset.",
    "trade rumors": "Trade-rumor data is unavailable in the current dataset.",
    "current NBA news": "Current NBA news is unavailable in the current dataset.",
}

UNAVAILABLE_CONSTRAINT_KEYWORDS = {
    "salary": ["salary", "salary cap", "cap space", "payroll"],
    "contracts": ["contract", "contracts", "expiring"],
    "injuries": ["injury", "injuries", "injured", "health status"],
    "trade rumors": ["trade rumor", "trade rumors", "rumor", "rumors"],
    "current NBA news": ["current news", "latest news", "today", "breaking news"],
}


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return fallback


def _find_team_from_query(user_query: str, prepared_data: PreparedNBAData) -> str | None:
    return find_team_in_query(user_query, prepared_data.teams)


def _parse_goal(user_query: str, filters: dict[str, Any]) -> str:
    if filters.get("goal") is not None:
        return str(filters["goal"])

    query = user_query.lower()
    for goal in GOAL_KEYWORDS:
        if goal in query:
            return goal
    return DEFAULT_REQUEST.goal


def _parse_from_text(patterns: list[str], user_query: str, fallback: int) -> int:
    for pattern in patterns:
        match = re.search(pattern, user_query, flags=re.IGNORECASE)
        if match:
            return _coerce_int(match.group(1), fallback)
    return fallback


def parse_query_deterministic(
    user_query: str,
    filters: dict[str, Any] | None,
    prepared_data: PreparedNBAData,
    warnings: list[str] | None = None,
) -> AnalysisRequest:
    """Parse a user query with deterministic fallback logic only."""

    filters = filters or {}
    warnings = warnings if warnings is not None else []
    parsed_fields = parse_user_query(
        user_query=user_query,
        defaults=filters,
        teams_df=prepared_data.teams,
        use_llm=False,
    )

    if not (filters.get("team_name") or filters.get("team")):
        if parsed_fields["team"]:
            warnings.append(f"Team filter missing; inferred team from query: {parsed_fields['team']}.")
        else:
            warnings.append(
                f"Team filter missing and no team was found in the query; defaulted to {DEFAULT_REQUEST.team_name}."
            )

    if filters.get("goal") is None:
        if parsed_fields["goal"]:
            warnings.append(f"Goal filter missing; inferred goal from query: {parsed_fields['goal']}.")
        else:
            warnings.append("Goal filter missing; no explicit goal was inferred.")

    missing_defaults = [
        ("top_k", DEFAULT_REQUEST.top_k),
        ("recent_games", DEFAULT_REQUEST.recent_games),
        ("min_games", DEFAULT_REQUEST.min_games),
        ("min_avg_minutes", DEFAULT_REQUEST.min_avg_minutes),
    ]
    for field, default_value in missing_defaults:
        if filters.get(field) is None:
            parsed_value = parsed_fields[field]
            if parsed_value == default_value:
                warnings.append(f"{field} missing; defaulted to {default_value:g}.")
            else:
                warnings.append(f"{field} missing; inferred from query: {parsed_value:g}.")

    if filters.get("exclude_current_team") is None:
        warnings.append(
            "exclude_current_team missing; "
            f"defaulted to {DEFAULT_REQUEST.exclude_current_team}."
        )

    if filters.get("ranking_mode") is None:
        warnings.append(
            f"ranking_mode missing; defaulted to {DEFAULT_REQUEST.ranking_mode}."
        )

    return _analysis_request_from_fields(parsed_fields)


def build_agent_plan(parsed_query: AnalysisRequest) -> list[str]:
    """Return the deterministic plan used by the orchestration layer."""

    return plan_tools(
        parsed_query,
        available_tools=[
            "Tool A: Team Need Diagnosis",
            "Tool B: Player Strength Representation",
            "Tool C: Fit Ranking",
        ],
        use_llm=False,
    )


def _analysis_request_from_fields(parsed_fields: dict[str, Any]) -> AnalysisRequest:
    return AnalysisRequest(
        team_name=str(parsed_fields["team"]),
        goal=str(parsed_fields.get("goal") or ""),
        top_k=_coerce_int(parsed_fields.get("top_k"), DEFAULT_REQUEST.top_k),
        recent_games=_coerce_int(
            parsed_fields.get("recent_games"), DEFAULT_REQUEST.recent_games
        ),
        min_games=_coerce_int(parsed_fields.get("min_games"), DEFAULT_REQUEST.min_games),
        min_avg_minutes=_coerce_float(
            parsed_fields.get("min_avg_minutes"), DEFAULT_REQUEST.min_avg_minutes
        ),
        exclude_current_team=_coerce_bool(
            parsed_fields.get("exclude_current_team"),
            DEFAULT_REQUEST.exclude_current_team,
        ),
        ranking_mode=str(parsed_fields.get("ranking_mode") or DEFAULT_REQUEST.ranking_mode),
        unavailable_constraints=tuple(parsed_fields.get("unavailable_constraints") or ()),
    )


def _build_final_summary(
    parsed_query: AnalysisRequest,
    team_name: str,
    need_df,
    ranked_df,
) -> str:
    top_needs = need_df.sort_values("need_weight", ascending=False).head(3)
    need_labels = top_needs["label"].tolist()
    player_names = ranked_df["PLAYER_NAME"].head(parsed_query.top_k).tolist()

    goal_text = parsed_query.goal if parsed_query.goal else "no explicit goal"
    players_text = ", ".join(player_names) if player_names else "no eligible players"

    return (
        f"Placeholder scouting summary: {team_name} was evaluated for {goal_text}. "
        f"Tool A identified the main statistical needs as {', '.join(need_labels)}. "
        f"Tool C's top statistical fits are {players_text} "
        f"using ranking mode '{parsed_query.ranking_mode}'. "
        "This summary is deterministic and does not include salary, contract, injury, "
        "or trade-rumor data."
    )


def _trace_step(
    step_number: int,
    title: str,
    short_description: str,
    key_outputs: dict[str, Any],
    status: str = "complete",
) -> TraceStep:
    return TraceStep(
        step_number=step_number,
        title=title,
        short_description=short_description,
        status=status,
        key_outputs=key_outputs,
    )


def _extend_warnings(warnings: list[str], new_warnings: list[str]) -> None:
    for warning in new_warnings:
        if warning not in warnings:
            warnings.append(warning)


def _detect_unavailable_constraints(user_query: str) -> list[str]:
    query = user_query.lower()
    constraints = []
    for label, keywords in UNAVAILABLE_CONSTRAINT_KEYWORDS.items():
        if any(keyword in query for keyword in keywords):
            constraints.append(label)
    return constraints


def _add_unavailable_constraint_warnings(
    warnings: list[str], unavailable_constraints: tuple[str, ...]
) -> None:
    for constraint in unavailable_constraints:
        warning = UNAVAILABLE_CONSTRAINT_WARNINGS.get(constraint)
        if warning and warning not in warnings:
            warnings.append(warning)


def run_roster_agent(
    user_query: str,
    filters: dict | None = None,
    data_dir: str = "data/raw",
    use_llm: bool = False,
) -> AgentResult:
    """Run the deterministic roster-upgrade pipeline end to end."""

    warnings: list[str] = []
    trace_steps: list[TraceStep] = []

    filters = filters or {}

    if "salary" in user_query.lower():
        warnings.append("Salary data is unavailable in the current dataset.")

    raw_data = load_raw_data(data_dir)
    prepared_data = prepare_data(raw_data)

    trace_steps.append(
        _trace_step(
            1,
            "User Query",
            "Original request submitted by the user.",
            {"user_query": user_query},
        )
    )

    if use_llm:
        if "_parsed_fields" in filters:
            parsed_query = _analysis_request_from_fields(filters["_parsed_fields"])
        else:
            parsed_fields = parse_user_query(
                user_query=user_query,
                defaults=filters,
                teams_df=prepared_data.teams,
                use_llm=True,
            )
            parsed_query = _analysis_request_from_fields(parsed_fields)
        _extend_warnings(warnings, get_llm_status().warnings)
        _extend_warnings(warnings, get_parser_warnings())
    elif "_parsed_fields" in filters:
        parsed_query = _analysis_request_from_fields(filters["_parsed_fields"])
    else:
        parsed_query = parse_query_deterministic(
            user_query, filters, prepared_data, warnings=warnings
        )

    _add_unavailable_constraint_warnings(warnings, parsed_query.unavailable_constraints)

    try:
        team_id = find_team_id(parsed_query.team_name, prepared_data.team_lookup)
    except ValueError:
        inferred_team = normalize_team_value(
            _find_team_from_query(user_query, prepared_data),
            prepared_data.teams,
            None,
        )
        fallback_team = inferred_team or DEFAULT_REQUEST.team_name
        warnings.append(
            f"Could not resolve team '{parsed_query.team_name}'; "
            f"using {fallback_team} instead."
        )
        parsed_query = AnalysisRequest(
            team_name=fallback_team,
            goal=parsed_query.goal,
            top_k=parsed_query.top_k,
            recent_games=parsed_query.recent_games,
            min_games=parsed_query.min_games,
            min_avg_minutes=parsed_query.min_avg_minutes,
            exclude_current_team=parsed_query.exclude_current_team,
            ranking_mode=parsed_query.ranking_mode,
            season=parsed_query.season,
            unavailable_constraints=parsed_query.unavailable_constraints,
        )
        team_id = find_team_id(parsed_query.team_name, prepared_data.team_lookup)
    resolved_team_name = prepared_data.team_name_map[team_id]
    season = (
        prepared_data.default_season
        if parsed_query.season is None
        else parsed_query.season
    )

    trace_steps.append(
        _trace_step(
            2,
            "Parsed Query",
            "Deterministic fallback parser resolved user intent and filters.",
            {
                "team_name": resolved_team_name,
                "team_id": team_id,
                "season": season,
                "goal": parsed_query.goal,
                "top_k": parsed_query.top_k,
                "recent_games": parsed_query.recent_games,
                "min_games": parsed_query.min_games,
                "min_avg_minutes": parsed_query.min_avg_minutes,
                "exclude_current_team": parsed_query.exclude_current_team,
                "ranking_mode": parsed_query.ranking_mode,
                "unavailable_constraints": list(parsed_query.unavailable_constraints),
            },
        )
    )

    available_tools = [
        "Tool A: Team Need Diagnosis",
        "LLM Need Reasoning",
        "Tool B: Player Strength Representation",
        "Tool C: Fit Ranking",
        "Sensitivity / Robustness Check",
    ]
    agent_plan = plan_tools(parsed_query, available_tools, use_llm=use_llm)
    if use_llm:
        _extend_warnings(warnings, get_llm_status().warnings)
        _extend_warnings(warnings, get_planner_warnings())

    trace_steps.append(
        _trace_step(
            3,
            "Agent Plan",
            "Ordered plan for Tool A, Tool B, and Tool C execution.",
            {"plan": agent_plan},
        )
    )

    need_df = diagnose_team_needs(
        prepared_data,
        team_id=team_id,
        season=season,
        recent_games=parsed_query.recent_games,
        goal=parsed_query.goal,
    )
    trace_steps.append(
        _trace_step(
            4,
            "Tool A: Team Need Diagnosis",
            "Computed recent team z-score needs against league context.",
            {
                "rows": len(need_df),
                "top_needs": need_df[["label", "need_weight"]]
                .head(3)
                .to_dict(orient="records"),
            },
        )
    )

    need_reasoning = reason_need_weights(
        parsed_query,
        need_df,
        allowed_metrics=CORE_METRIC_LABELS,
        use_llm=use_llm,
    )
    trace_steps.append(
        _trace_step(
            5,
            "LLM Need Reasoning",
            "Validated tactical multipliers were applied to Tool A need weights.",
            {
                "used_fallback": need_reasoning.used_fallback,
                "metric_multipliers": need_reasoning.metric_multipliers,
            },
        )
    )

    player_strength_df = build_player_strengths(
        prepared_data,
        season=season,
        min_games=parsed_query.min_games,
        min_avg_minutes=parsed_query.min_avg_minutes,
        exclude_current_team=parsed_query.exclude_current_team,
        current_team_id=team_id,
    )
    trace_steps.append(
        _trace_step(
            6,
            "Tool B: Player Strength Representation",
            "Built candidate player strength vectors from box-score features.",
            {
                "candidate_count": len(player_strength_df),
                "filters": {
                    "min_games": parsed_query.min_games,
                    "min_avg_minutes": parsed_query.min_avg_minutes,
                    "exclude_current_team": parsed_query.exclude_current_team,
                    "ranking_mode": parsed_query.ranking_mode,
                },
            },
        )
    )

    ranked_df = rank_players_by_fit(
        need_reasoning.adjusted_need_df,
        player_strength_df,
        team_id=team_id,
        top_k=parsed_query.top_k,
        exclude_current_team=False,
    )
    trace_steps.append(
        _trace_step(
            7,
            "Tool C: Fit Ranking",
            "Ranked candidates by matching adjusted needs to Tool B strengths.",
            {
                "top_k": parsed_query.top_k,
                "top_players": ranked_df[
                    ["PLAYER_NAME", "CURRENT_TEAM", "fit_score", "best_match"]
                ].to_dict(orient="records"),
            },
        )
    )

    sensitivity = run_sensitivity_check(
        need_reasoning.adjusted_need_df,
        player_strength_df,
        ranked_df,
        team_id=team_id,
        top_k=parsed_query.top_k,
    )
    trace_steps.append(
        _trace_step(
            8,
            "Sensitivity / Robustness Check",
            "Compared top recommendations after a small need-weight perturbation.",
            {
                "stability_label": sensitivity.stability_label,
                "top_k_overlap": sensitivity.top_k_overlap,
            },
        )
    )

    scouting_summary = summarize_scouting_report(
        parsed_query,
        agent_plan,
        need_df,
        need_reasoning.adjusted_need_df,
        need_reasoning,
        ranked_df,
        sensitivity,
        use_llm=use_llm,
    )
    trace_steps.append(
        _trace_step(
            9,
            "Final Scouting Summary",
            "Grounded scouting summary from computed pipeline outputs.",
            {"summary": scouting_summary.executive_summary},
        )
    )

    return AgentResult(
        user_query=user_query,
        parsed_query=parsed_query,
        agent_plan=agent_plan,
        need_df=need_df,
        need_reasoning=need_reasoning,
        player_strength_df=player_strength_df,
        ranked_df=ranked_df,
        sensitivity=sensitivity,
        scouting_summary=scouting_summary,
        final_summary=scouting_summary.executive_summary,
        warnings=warnings,
        trace_steps=trace_steps,
    )


class NBARosterUpgradeAgent:
    """Small wrapper for callers that prefer an object-oriented interface."""

    def __init__(self, data_dir: str = "data/raw", use_llm: bool = False):
        self.data_dir = data_dir
        self.use_llm = use_llm

    def run(self, user_query: str, filters: dict | None = None) -> AgentResult:
        return run_roster_agent(
            user_query=user_query,
            filters=filters or {},
            data_dir=self.data_dir,
            use_llm=self.use_llm,
        )
