"""Top-level deterministic orchestration for the roster-upgrade pipeline."""

from __future__ import annotations

import re
from typing import Any

from nba_agent.data.loader import load_raw_data
from nba_agent.data.preprocessing import find_team_id, prepare_data
from nba_agent.schemas import AgentResult, AnalysisRequest, PreparedNBAData, TraceStep
from nba_agent.tools.fit_ranking import rank_players_by_fit
from nba_agent.tools.need_diagnosis import diagnose_team_needs
from nba_agent.tools.player_strength import build_player_strengths


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
    query = user_query.lower()
    teams = prepared_data.teams.copy()

    candidate_names: list[str] = []
    for _, row in teams.iterrows():
        for column in ["TEAM_NAME_FULL", "NICKNAME", "ABBREVIATION", "CITY"]:
            value = row.get(column)
            if value is not None and str(value).strip():
                candidate_names.append(str(value).strip())

    for name in sorted(set(candidate_names), key=len, reverse=True):
        pattern = r"\b" + re.escape(name.lower()) + r"\b"
        if re.search(pattern, query):
            return name

    return None


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

    explicit_team = filters.get("team_name") or filters.get("team")
    inferred_team = _find_team_from_query(user_query, prepared_data)
    if explicit_team:
        team_name = str(explicit_team)
    elif inferred_team:
        team_name = inferred_team
        warnings.append(f"Team filter missing; inferred team from query: {team_name}.")
    else:
        team_name = DEFAULT_REQUEST.team_name
        warnings.append(
            f"Team filter missing and no team was found in the query; defaulted to {team_name}."
        )

    if filters.get("goal") is None:
        inferred_goal = _parse_goal(user_query, filters)
        if inferred_goal:
            warnings.append(f"Goal filter missing; inferred goal from query: {inferred_goal}.")
        else:
            warnings.append("Goal filter missing; no explicit goal was inferred.")
    else:
        inferred_goal = _parse_goal(user_query, filters)

    top_k = _coerce_int(
        filters.get("top_k"),
        _parse_from_text(
            [r"\btop\s+(\d+)\b", r"\brecommend\s+(\d+)\b"],
            user_query,
            DEFAULT_REQUEST.top_k,
        ),
    )
    if filters.get("top_k") is None:
        if top_k == DEFAULT_REQUEST.top_k:
            warnings.append(f"top_k missing; defaulted to {DEFAULT_REQUEST.top_k}.")
        else:
            warnings.append(f"top_k missing; inferred from query: {top_k}.")

    recent_games = _coerce_int(
        filters.get("recent_games"),
        _parse_from_text(
            [r"\blast\s+(\d+)\s+games?\b", r"\brecent\s+(\d+)\s+games?\b"],
            user_query,
            DEFAULT_REQUEST.recent_games,
        ),
    )
    if filters.get("recent_games") is None:
        if recent_games == DEFAULT_REQUEST.recent_games:
            warnings.append(
                f"recent_games missing; defaulted to {DEFAULT_REQUEST.recent_games}."
            )
        else:
            warnings.append(f"recent_games missing; inferred from query: {recent_games}.")

    min_games = _coerce_int(
        filters.get("min_games"),
        _parse_from_text(
            [r"\bat least\s+(\d+)\s+games?\b", r"\bminimum\s+(\d+)\s+games?\b"],
            user_query,
            DEFAULT_REQUEST.min_games,
        ),
    )
    if filters.get("min_games") is None:
        if min_games == DEFAULT_REQUEST.min_games:
            warnings.append(
                f"min_games missing; defaulted to {DEFAULT_REQUEST.min_games}."
            )
        else:
            warnings.append(f"min_games missing; inferred from query: {min_games}.")

    min_avg_minutes = _coerce_float(
        filters.get("min_avg_minutes"),
        float(
            _parse_from_text(
                [
                    r"\bat least\s+(\d+)\s+average minutes?\b",
                    r"\b(\d+)\s+average minutes?\b",
                    r"\bminimum\s+(\d+)\s+minutes?\b",
                ],
                user_query,
                int(DEFAULT_REQUEST.min_avg_minutes),
            )
        ),
    )
    if filters.get("min_avg_minutes") is None:
        if min_avg_minutes == DEFAULT_REQUEST.min_avg_minutes:
            warnings.append(
                "min_avg_minutes missing; "
                f"defaulted to {DEFAULT_REQUEST.min_avg_minutes:g}."
            )
        else:
            warnings.append(
                f"min_avg_minutes missing; inferred from query: {min_avg_minutes:g}."
            )

    exclude_current_team = _coerce_bool(
        filters.get("exclude_current_team"),
        DEFAULT_REQUEST.exclude_current_team,
    )
    if filters.get("exclude_current_team") is None:
        warnings.append(
            "exclude_current_team missing; "
            f"defaulted to {DEFAULT_REQUEST.exclude_current_team}."
        )

    ranking_mode = str(filters.get("ranking_mode") or DEFAULT_REQUEST.ranking_mode)
    if filters.get("ranking_mode") is None:
        warnings.append(
            f"ranking_mode missing; defaulted to {DEFAULT_REQUEST.ranking_mode}."
        )

    season = filters.get("season")
    parsed_season = _coerce_int(season, 0) if season is not None else None

    return AnalysisRequest(
        team_name=team_name,
        goal=_parse_goal(user_query, filters),
        top_k=max(top_k, 1),
        recent_games=max(recent_games, 1),
        min_games=max(min_games, 1),
        min_avg_minutes=max(min_avg_minutes, 0.0),
        exclude_current_team=exclude_current_team,
        ranking_mode=ranking_mode,
        season=parsed_season,
    )


def build_agent_plan(parsed_query: AnalysisRequest) -> list[str]:
    """Return the deterministic plan used by the orchestration layer."""

    return [
        f"Resolve team '{parsed_query.team_name}' and season.",
        f"Run Tool A on the last {parsed_query.recent_games} team games.",
        (
            "Run Tool B to build player strength vectors with "
            f"min_games={parsed_query.min_games} and "
            f"min_avg_minutes={parsed_query.min_avg_minutes:g}."
        ),
        f"Run Tool C to rank the top {parsed_query.top_k} statistical fits.",
        f"Use ranking mode: {parsed_query.ranking_mode}.",
        "Return a grounded placeholder scouting summary.",
    ]


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

    if use_llm:
        warnings.append(
            "LLM mode is not implemented yet; deterministic fallback mode was used."
        )

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

    parsed_query = parse_query_deterministic(
        user_query, filters, prepared_data, warnings=warnings
    )
    try:
        team_id = find_team_id(parsed_query.team_name, prepared_data.team_lookup)
    except ValueError:
        inferred_team = _find_team_from_query(user_query, prepared_data)
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
            },
        )
    )

    agent_plan = build_agent_plan(parsed_query)
    trace_steps.append(
        _trace_step(
            3,
            "Agent Plan",
            "Deterministic plan for Tool A, Tool B, and Tool C execution.",
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
            5,
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
        need_df,
        player_strength_df,
        team_id=team_id,
        top_k=parsed_query.top_k,
        exclude_current_team=False,
    )
    trace_steps.append(
        _trace_step(
            6,
            "Tool C: Fit Ranking",
            "Ranked candidates by matching Tool A needs to Tool B strengths.",
            {
                "top_k": parsed_query.top_k,
                "top_players": ranked_df[
                    ["PLAYER_NAME", "CURRENT_TEAM", "fit_score", "best_match"]
                ].to_dict(orient="records"),
            },
        )
    )

    final_summary = _build_final_summary(
        parsed_query,
        resolved_team_name,
        need_df,
        ranked_df,
    )
    trace_steps.append(
        _trace_step(
            7,
            "Final Scouting Summary",
            "Placeholder grounded summary from deterministic pipeline outputs.",
            {"summary": final_summary},
        )
    )

    return AgentResult(
        user_query=user_query,
        parsed_query=parsed_query,
        agent_plan=agent_plan,
        need_df=need_df,
        player_strength_df=player_strength_df,
        ranked_df=ranked_df,
        final_summary=final_summary,
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
