"""Optional LLM query parsing with deterministic validation and fallback."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from nba_agent.llm.client import call_llm_json
from nba_agent.llm.prompts import QUERY_PARSER_PROMPT


DEFAULT_PARSED_QUERY = {
    "team": "Warriors",
    "goal": "",
    "top_k": 5,
    "recent_games": 10,
    "min_games": 15,
    "min_avg_minutes": 15.0,
    "exclude_current_team": True,
    "ranking_mode": "Best Talent",
    "unavailable_constraints": [],
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

RANKING_MODES = {"Best Talent", "Realistic Fit", "Hidden Gems"}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
}

TEAM_ALIAS_OVERRIDES = {
    "gsw": "Golden State Warriors",
    "golden state": "Golden State Warriors",
    "lal": "Los Angeles Lakers",
    "la lakers": "Los Angeles Lakers",
    "bos": "Boston Celtics",
    "nyk": "New York Knicks",
    "new york": "New York Knicks",
    "mia": "Miami Heat",
    "den": "Denver Nuggets",
    "dal": "Dallas Mavericks",
    "mavs": "Dallas Mavericks",
    "phx": "Phoenix Suns",
    "pho": "Phoenix Suns",
    "mil": "Milwaukee Bucks",
    "phi": "Philadelphia 76ers",
    "sixers": "Philadelphia 76ers",
    "76ers": "Philadelphia 76ers",
    "lac": "Los Angeles Clippers",
    "la clippers": "Los Angeles Clippers",
    "bkn": "Brooklyn Nets",
    "brk": "Brooklyn Nets",
    "chi": "Chicago Bulls",
    "cle": "Cleveland Cavaliers",
    "cavs": "Cleveland Cavaliers",
    "min": "Minnesota Timberwolves",
    "wolves": "Minnesota Timberwolves",
    "okc": "Oklahoma City Thunder",
    "por": "Portland Trail Blazers",
    "blazers": "Portland Trail Blazers",
    "trail blazers": "Portland Trail Blazers",
    "sac": "Sacramento Kings",
    "sas": "San Antonio Spurs",
    "sa spurs": "San Antonio Spurs",
    "tor": "Toronto Raptors",
    "uta": "Utah Jazz",
    "hou": "Houston Rockets",
    "atl": "Atlanta Hawks",
    "cha": "Charlotte Hornets",
    "det": "Detroit Pistons",
    "ind": "Indiana Pacers",
    "mem": "Memphis Grizzlies",
    "nop": "New Orleans Pelicans",
    "no pelicans": "New Orleans Pelicans",
    "orl": "Orlando Magic",
    "was": "Washington Wizards",
    "wsh": "Washington Wizards",
}

UNAVAILABLE_CONSTRAINT_KEYWORDS = {
    "salary": ["salary", "salary cap", "cap space", "payroll"],
    "contracts": ["contract", "contracts", "expiring"],
    "injuries": ["injury", "injuries", "injured", "health status"],
    "trade rumors": ["trade rumor", "trade rumors", "rumor", "rumors"],
    "current NBA news": ["current news", "latest news", "today", "breaking news"],
}

_PARSER_WARNINGS: list[str] = []


def get_parser_warnings() -> list[str]:
    """Return warnings from the most recent parser calls."""

    return list(_PARSER_WARNINGS)


def parse_user_query(
    user_query: str,
    defaults: dict,
    teams_df,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Parse a roster-upgrade request into validated structured fields."""

    _PARSER_WARNINGS.clear()
    defaults = defaults or {}
    fallback = _deterministic_parse(user_query, defaults, teams_df)

    if not use_llm:
        return fallback

    sentinel = {"_fallback": True}
    messages = [
        {"role": "system", "content": QUERY_PARSER_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user_query": user_query,
                    "defaults": _json_safe_defaults(defaults),
                    "valid_teams": _team_names(teams_df),
                    "valid_team_aliases": sorted(_team_aliases(teams_df)),
                    "ranking_modes": sorted(RANKING_MODES),
                    "required_fields": list(DEFAULT_PARSED_QUERY.keys()),
                },
                ensure_ascii=True,
            ),
        },
    ]
    parsed = call_llm_json(messages, fallback=sentinel)
    if parsed == sentinel:
        return fallback
    if not isinstance(parsed, dict):
        _add_warning("LLM parser returned a non-object JSON value; deterministic fallback was used.")
        return fallback

    validated = _validate_parsed_fields(parsed, fallback, user_query, teams_df)
    if validated == fallback and parsed != fallback:
        _add_warning("LLM parser output was normalized with deterministic query extraction.")
    return validated


def normalize_team_value(team: Any, teams_df, fallback: str | None = None) -> str | None:
    """Normalize a full name, nickname, abbreviation, or common alias to a full team name."""

    if _invalid_team_value(team):
        return fallback

    team_text = str(team).strip()
    aliases = _team_aliases(teams_df)
    key = _normalize_alias_key(team_text)
    if key in aliases:
        return aliases[key]

    for alias, canonical in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if key and (key == alias or key in alias):
            return canonical

    return fallback


def find_team_in_query(user_query: str, teams_df) -> str | None:
    """Return a canonical team name found in natural language, if present."""

    return _find_team_from_query(user_query, teams_df)


def _deterministic_parse(user_query: str, defaults: dict, teams_df) -> dict[str, Any]:
    parsed = dict(DEFAULT_PARSED_QUERY)

    inferred_team = _find_team_from_query(user_query, teams_df)
    default_team = defaults.get("team") or defaults.get("team_name")
    parsed["team"] = str(inferred_team or default_team or parsed["team"])

    inferred_goal = _find_goal_from_query(user_query)
    default_goal = defaults.get("goal")
    parsed["goal"] = str(
        inferred_goal or (default_goal if default_goal is not None else "")
    )

    parsed["top_k"] = _coerce_int(
        _parse_from_text(
            [r"\btop\s+(\d+|[a-z]+)\b", r"\brecommend\s+(\d+|[a-z]+)\b"],
            user_query,
            defaults.get("top_k", 5),
        ),
        5,
    )
    parsed["recent_games"] = _coerce_int(
        _parse_from_text(
            [r"\blast\s+(\d+)\s+games?\b", r"\brecent\s+(\d+)\s+games?\b"],
            user_query,
            defaults.get("recent_games", 10),
        ),
        10,
    )
    parsed["min_games"] = _coerce_int(
        _parse_from_text(
            [r"\bat least\s+(\d+|[a-z]+)\s+games?\b", r"\bminimum\s+(\d+|[a-z]+)\s+games?\b"],
            user_query,
            defaults.get("min_games", 15),
        ),
        15,
    )
    parsed["min_avg_minutes"] = _coerce_float(
        float(
            _parse_from_text(
                [
                    r"\bat least\s+(\d+)\s+average minutes?\b",
                    r"\bat least\s+(\d+)\s+avg minutes?\b",
                    r"\b(\d+)\s+average minutes?\b",
                    r"\b(\d+)\s+avg minutes?\b",
                    r"\bminimum\s+(\d+)\s+minutes?\b",
                ],
                user_query,
                int(_coerce_float(defaults.get("min_avg_minutes"), 15.0)),
            )
        ),
        15.0,
    )
    parsed["exclude_current_team"] = _coerce_bool(
        defaults.get("exclude_current_team"), True
    )
    parsed["ranking_mode"] = _validate_ranking_mode(
        defaults.get("ranking_mode") or "Best Talent"
    )
    parsed["unavailable_constraints"] = _unavailable_constraints(user_query)

    return _validate_parsed_fields(parsed, DEFAULT_PARSED_QUERY, user_query, teams_df)


def _validate_parsed_fields(
    parsed: dict[str, Any], fallback: dict[str, Any], user_query: str, teams_df
) -> dict[str, Any]:
    validated = dict(fallback)

    validated["team"] = _validate_team(parsed.get("team"), fallback["team"], user_query, teams_df)
    if not validated["team"]:
        validated["team"] = fallback["team"]

    goal = parsed.get("goal", fallback["goal"])
    validated["goal"] = str(goal).strip() if goal is not None else fallback["goal"]

    validated["top_k"] = max(_coerce_int(parsed.get("top_k"), fallback["top_k"]), 1)
    validated["recent_games"] = max(
        _coerce_int(parsed.get("recent_games"), fallback["recent_games"]), 1
    )
    validated["min_games"] = max(
        _coerce_int(parsed.get("min_games"), fallback["min_games"]), 1
    )
    validated["min_avg_minutes"] = max(
        _coerce_float(parsed.get("min_avg_minutes"), fallback["min_avg_minutes"]), 0.0
    )
    validated["exclude_current_team"] = _coerce_bool(
        parsed.get("exclude_current_team"), fallback["exclude_current_team"]
    )
    validated["ranking_mode"] = _validate_ranking_mode(
        parsed.get("ranking_mode") or fallback["ranking_mode"]
    )
    validated["unavailable_constraints"] = _validate_unavailable_constraints(
        parsed.get("unavailable_constraints"), user_query
    )

    return validated


def _validate_team(team: Any, fallback: str, user_query: str, teams_df) -> str:
    query_team = _find_team_from_query(user_query, teams_df)
    normalized_fallback = normalize_team_value(query_team or fallback, teams_df, fallback)
    normalized = normalize_team_value(team, teams_df, normalized_fallback)

    if normalized:
        return normalized

    if not _invalid_team_value(team):
        _add_warning("LLM parser returned an unknown team; deterministic query team was used.")
    return normalized_fallback or fallback


def _team_names(teams_df) -> list[str]:
    if teams_df is None or not isinstance(teams_df, pd.DataFrame):
        return []
    names = []
    for _, row in teams_df.iterrows():
        full_name = _full_team_name(row)
        if full_name:
            names.append(full_name)
    return sorted(set(names))


def _team_aliases(teams_df) -> dict[str, str]:
    if teams_df is None or not isinstance(teams_df, pd.DataFrame):
        return {}

    aliases: dict[str, str] = {}
    for _, row in teams_df.iterrows():
        canonical = _full_team_name(row)
        if not canonical:
            continue
        for column in ["TEAM_NAME_FULL", "NICKNAME", "ABBREVIATION", "CITY"]:
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                aliases[_normalize_alias_key(value)] = canonical
        aliases[_normalize_alias_key(canonical)] = canonical

    for alias, canonical in TEAM_ALIAS_OVERRIDES.items():
        if canonical in set(aliases.values()):
            aliases[_normalize_alias_key(alias)] = canonical
    return aliases


def _full_team_name(row) -> str:
    full_name = row.get("TEAM_NAME_FULL")
    if pd.notna(full_name) and str(full_name).strip():
        return str(full_name).strip()
    city = str(row.get("CITY", "") or "").strip()
    nickname = str(row.get("NICKNAME", "") or "").strip()
    return f"{city} {nickname}".strip()


def _find_team_from_query(user_query: str, teams_df) -> str | None:
    query = _normalize_alias_key(user_query)
    for alias, canonical in sorted(_team_aliases(teams_df).items(), key=lambda item: len(item[0]), reverse=True):
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, query):
            return canonical
    return None


def _find_goal_from_query(user_query: str) -> str:
    query = user_query.lower()
    for goal in GOAL_KEYWORDS:
        if goal in query:
            return goal
    return ""


def _parse_from_text(patterns: list[str], user_query: str, fallback: int) -> int:
    for pattern in patterns:
        match = re.search(pattern, user_query, flags=re.IGNORECASE)
        if match:
            return _coerce_int(match.group(1), fallback)
    return fallback


def _validate_ranking_mode(value: Any) -> str:
    text = str(value).strip() if value is not None else "Best Talent"
    for mode in RANKING_MODES:
        if text.lower() == mode.lower():
            return mode
    return "Best Talent"


def _unavailable_constraints(user_query: str) -> list[str]:
    query = user_query.lower()
    constraints = []
    for label, keywords in UNAVAILABLE_CONSTRAINT_KEYWORDS.items():
        if any(keyword in query for keyword in keywords):
            constraints.append(label)
    return constraints


def _validate_unavailable_constraints(value: Any, user_query: str) -> list[str]:
    query_constraints = set(_unavailable_constraints(user_query))
    if not isinstance(value, list):
        return sorted(query_constraints)

    allowed = set(UNAVAILABLE_CONSTRAINT_KEYWORDS)
    output = query_constraints
    for item in value:
        label = str(item).strip().lower()
        if label in allowed and any(
            keyword in user_query.lower()
            for keyword in UNAVAILABLE_CONSTRAINT_KEYWORDS[label]
        ):
            output.add(label)
    return sorted(output)


def _coerce_int(value: Any, fallback: int) -> int:
    if isinstance(value, str) and value.strip().lower() in NUMBER_WORDS:
        return NUMBER_WORDS[value.strip().lower()]
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


def _json_safe_defaults(defaults: dict) -> dict[str, Any]:
    return {
        key: value
        for key, value in defaults.items()
        if isinstance(value, (str, int, float, bool, list, dict, tuple)) or value is None
    }


def _add_warning(message: str) -> None:
    if message not in _PARSER_WARNINGS:
        _PARSER_WARNINGS.append(message)


def _normalize_alias_key(value: Any) -> str:
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _invalid_team_value(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    normalized = _normalize_alias_key(text)
    if not normalized:
        return True
    return normalized in {"unknown", "none", "null", "n/a", "na"}
