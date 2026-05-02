"""Shared schemas for the NBA roster-upgrade pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AnalysisRequest:
    """User-facing parameters for the roster-upgrade workflow."""

    team_name: str
    goal: str = ""
    top_k: int = 5
    recent_games: int = 10
    min_games: int = 15
    min_avg_minutes: float = 15.0
    exclude_current_team: bool = True
    ranking_mode: str = "Best Talent"
    season: int | None = None
    unavailable_constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class RawNBAData:
    """Raw CSV data loaded from the original dataset style."""

    teams: pd.DataFrame
    games: pd.DataFrame
    games_details: pd.DataFrame
    players: pd.DataFrame | None = None
    ranking: pd.DataFrame | None = None
    data_dir: Path | None = None


@dataclass(frozen=True)
class PreparedNBAData:
    """Cleaned and joined dataframes used by deterministic tools."""

    teams: pd.DataFrame
    games: pd.DataFrame
    games_details: pd.DataFrame
    player_game_df: pd.DataFrame
    team_game_df: pd.DataFrame
    team_name_map: dict[int, str]
    team_abbr_map: dict[int, str]
    team_lookup: dict[str, int]
    default_season: int


@dataclass(frozen=True)
class TraceStep:
    """Streamlit-friendly pipeline trace item."""

    step_number: int
    title: str
    short_description: str
    status: str
    key_outputs: dict[str, Any]


@dataclass(frozen=True)
class NeedReasoningResult:
    """LLM/fallback interpretation of Tool A needs before ranking."""

    tactical_interpretation: str
    metric_multipliers: dict[str, float]
    explanations: dict[str, str]
    adjusted_need_df: pd.DataFrame
    warnings: list[str]
    used_fallback: bool


@dataclass(frozen=True)
class SensitivityResult:
    """Robustness check for recommendation stability under small weight changes."""

    stability_label: str
    top_k_overlap: float
    original_top_players: list[str]
    perturbed_top_players: list[str]
    explanation: str
    rank_comparison_df: pd.DataFrame


@dataclass(frozen=True)
class AgentResult:
    """Structured result returned by the high-level roster agent."""

    user_query: str
    parsed_query: AnalysisRequest
    agent_plan: list[str]
    need_df: pd.DataFrame
    need_reasoning: NeedReasoningResult
    player_strength_df: pd.DataFrame
    ranked_df: pd.DataFrame
    sensitivity: SensitivityResult
    final_summary: str
    warnings: list[str]
    trace_steps: list[TraceStep]

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary while preserving dataframe objects."""

        return {
            "user_query": self.user_query,
            "parsed_query": self.parsed_query,
            "agent_plan": self.agent_plan,
            "need_df": self.need_df,
            "need_reasoning": self.need_reasoning,
            "player_strength_df": self.player_strength_df,
            "ranked_df": self.ranked_df,
            "sensitivity": self.sensitivity,
            "final_summary": self.final_summary,
            "warnings": self.warnings,
            "trace_steps": self.trace_steps,
        }
