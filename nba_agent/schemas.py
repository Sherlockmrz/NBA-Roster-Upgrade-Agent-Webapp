"""Shared schemas for the NBA roster-upgrade pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    season: int | None = None


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
