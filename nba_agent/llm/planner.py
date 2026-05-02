"""Optional LLM agent planning with deterministic order guarantees."""

from __future__ import annotations

from typing import Any


REQUIRED_TOOL_ORDER = [
    "Tool A: Team Need Diagnosis",
    "Tool B: Player Strength Representation",
    "Tool C: Fit Ranking",
]

_PLANNER_WARNINGS: list[str] = []


def get_planner_warnings() -> list[str]:
    """Return warnings from the most recent planner calls."""

    return list(_PLANNER_WARNINGS)


def plan_tools(
    parsed_query,
    available_tools,
    use_llm: bool = True,
) -> list[str]:
    """Return an ordered display-friendly agent plan.

    The LLM never controls Tool A / Tool B / Tool C order. The current app keeps
    the deterministic plan as the source of truth and leaves LLM planning as a
    future explanatory layer.
    """

    _PLANNER_WARNINGS.clear()
    return _deterministic_plan(parsed_query, available_tools)


def _deterministic_plan(parsed_query, available_tools) -> list[str]:
    query = _parsed_query_dict(parsed_query)
    team = query.get("team") or query.get("team_name") or "selected team"
    goal = query.get("goal") or "the requested roster goal"
    top_k = query.get("top_k", 5)
    recent_games = query.get("recent_games", 10)
    min_games = query.get("min_games", 15)
    min_avg_minutes = query.get("min_avg_minutes", 15)
    ranking_mode = query.get("ranking_mode", "Best Talent")

    return [
        f"Resolve team '{team}', goal '{goal}', and user filters.",
        f"Run Tool A: Team Need Diagnosis on the last {recent_games} team games.",
        (
            "Run Tool B: Player Strength Representation with "
            f"min_games={min_games} and min_avg_minutes={float(min_avg_minutes):g}."
        ),
        f"Run Tool C: Fit Ranking to return the top {top_k} statistical fits.",
        "Run Sensitivity / Robustness Check by perturbing adjusted need weights and comparing top recommendations.",
        f"Display ranking mode: {ranking_mode}.",
        "Surface unavailable constraints without inventing missing data.",
        "Return a grounded scouting summary from deterministic outputs.",
    ]


def _preserves_required_order(plan: list[str]) -> bool:
    joined = "\n".join(plan)
    positions = []
    for tool in REQUIRED_TOOL_ORDER:
        position = joined.find(tool)
        if position == -1:
            return False
        positions.append(position)
    return positions == sorted(positions)


def _parsed_query_dict(parsed_query) -> dict[str, Any]:
    if isinstance(parsed_query, dict):
        return dict(parsed_query)
    if hasattr(parsed_query, "__dataclass_fields__"):
        fields = parsed_query.__dataclass_fields__
        return {name: getattr(parsed_query, name) for name in fields}
    return {}


def _add_warning(message: str) -> None:
    if message not in _PLANNER_WARNINGS:
        _PLANNER_WARNINGS.append(message)
