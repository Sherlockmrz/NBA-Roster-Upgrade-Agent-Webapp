"""Preprocessing helpers for the original NBA box-score dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_agent.schemas import PreparedNBAData, RawNBAData


NUMERIC_BOX_SCORE_COLUMNS = [
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TO",
    "PF",
    "PTS",
    "PLUS_MINUS",
]


TEAM_GAME_AGG_COLUMNS = ["REB", "AST", "STL", "BLK", "FG3M", "FG3A", "PTS"]


def parse_minutes(value: object) -> float:
    """Parse NBA minutes strings like ``18:06`` into decimal minutes."""

    if pd.isna(value):
        return 0.0

    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return 0.0

    if ":" in text:
        try:
            minutes, seconds = text.split(":", maxsplit=1)
            return float(minutes) + float(seconds) / 60.0
        except (TypeError, ValueError):
            return 0.0

    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def build_team_lookup(teams: pd.DataFrame) -> dict[str, int]:
    """Build flexible lookup keys for team id resolution."""

    lookup: dict[str, int] = {}
    for _, row in teams.iterrows():
        keys = [
            row.get("TEAM_ID"),
            row.get("ABBREVIATION"),
            row.get("NICKNAME"),
            row.get("CITY"),
            row.get("TEAM_NAME_FULL"),
        ]
        for key in keys:
            if pd.notna(key) and str(key).strip():
                lookup[str(key).strip().lower()] = int(row["TEAM_ID"])
    return lookup


def find_team_id(team_text: str, team_lookup: dict[str, int]) -> int:
    """Resolve a user-provided team string to a TEAM_ID."""

    if team_text is None:
        raise ValueError("No team name provided.")

    key = str(team_text).strip().lower()
    if key in team_lookup:
        return team_lookup[key]

    for candidate, team_id in team_lookup.items():
        if key and key in candidate:
            return team_id

    raise ValueError(f"Could not match team name: {team_text}")


def prepare_data(raw_data: RawNBAData) -> PreparedNBAData:
    """Clean raw data and build reusable player-game and team-game tables."""

    teams = raw_data.teams.copy()
    games = raw_data.games.copy()
    games_details = raw_data.games_details.copy()

    games["GAME_DATE_EST"] = pd.to_datetime(games["GAME_DATE_EST"], errors="coerce")

    for column in NUMERIC_BOX_SCORE_COLUMNS:
        if column in games_details.columns:
            games_details[column] = pd.to_numeric(
                games_details[column], errors="coerce"
            ).fillna(0)

    games_details["MIN_FLOAT"] = games_details["MIN"].apply(parse_minutes)
    games_details = games_details[games_details["MIN_FLOAT"] > 0].copy()

    teams["TEAM_NAME_FULL"] = (
        teams["CITY"].fillna("") + " " + teams["NICKNAME"].fillna("")
    ).str.strip()

    team_name_map = dict(zip(teams["TEAM_ID"].astype(int), teams["TEAM_NAME_FULL"]))
    team_abbr_map = dict(zip(teams["TEAM_ID"].astype(int), teams["ABBREVIATION"]))
    team_lookup = build_team_lookup(teams)

    games_small = games[
        ["GAME_ID", "GAME_DATE_EST", "SEASON", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]
    ].copy()

    player_game_df = games_details.merge(
        games_small[["GAME_ID", "GAME_DATE_EST", "SEASON"]],
        on="GAME_ID",
        how="left",
    )

    team_game_df = (
        player_game_df.groupby(["GAME_ID", "TEAM_ID"], as_index=False)
        .agg({column: "sum" for column in TEAM_GAME_AGG_COLUMNS})
        .merge(games_small, on="GAME_ID", how="left")
    )

    team_game_df["FG3_PCT"] = np.where(
        team_game_df["FG3A"] > 0,
        team_game_df["FG3M"] / team_game_df["FG3A"],
        0,
    )

    return PreparedNBAData(
        teams=teams,
        games=games,
        games_details=games_details,
        player_game_df=player_game_df,
        team_game_df=team_game_df,
        team_name_map=team_name_map,
        team_abbr_map=team_abbr_map,
        team_lookup=team_lookup,
        default_season=int(games["SEASON"].max()),
    )


def preprocess(raw_data: RawNBAData) -> PreparedNBAData:
    """Backward-compatible alias for ``prepare_data``."""

    return prepare_data(raw_data)
