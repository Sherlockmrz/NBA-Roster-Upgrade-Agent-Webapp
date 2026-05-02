"""Load the original NBA roster-upgrade CSV dataset.

This module only reads the dataset style already present in ``data/raw``.
Salary, contract, injury, and trade-rumor datasets are intentionally out of
scope for the current project.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nba_agent.schemas import RawNBAData


DEFAULT_DATA_DIR = Path("data/raw")


REQUIRED_RAW_FILES = {
    "teams": "teams.csv",
    "games": "games.csv",
    "games_details": "games_details.csv",
}


OPTIONAL_RAW_FILES = {
    "players": "players.csv",
    "ranking": "ranking.csv",
}


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Return a repo-relative or caller-provided raw data directory."""

    return Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR


def load_raw_data(data_dir: str | Path | None = None) -> RawNBAData:
    """Load raw NBA CSV files from a configurable directory.

    Parameters
    ----------
    data_dir:
        Directory containing the original CSV files. Defaults to ``data/raw``.

    Returns
    -------
    RawNBAData
        Container with required dataframes plus optional ``players`` and
        ``ranking`` frames when those files exist.
    """

    raw_dir = resolve_data_dir(data_dir)
    missing = [
        filename
        for filename in REQUIRED_RAW_FILES.values()
        if not (raw_dir / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing required raw data files in {raw_dir}: {', '.join(missing)}"
        )

    frames = {
        name: pd.read_csv(raw_dir / filename, low_memory=False)
        for name, filename in REQUIRED_RAW_FILES.items()
    }

    optional_frames: dict[str, pd.DataFrame | None] = {}
    for name, filename in OPTIONAL_RAW_FILES.items():
        path = raw_dir / filename
        optional_frames[name] = (
            pd.read_csv(path, low_memory=False) if path.exists() else None
        )

    return RawNBAData(
        teams=frames["teams"],
        games=frames["games"],
        games_details=frames["games_details"],
        players=optional_frames["players"],
        ranking=optional_frames["ranking"],
        data_dir=raw_dir,
    )
