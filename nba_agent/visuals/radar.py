"""Compact radar visualization helpers for player preview cards."""

from __future__ import annotations

from html import escape
import math
from typing import Any

import pandas as pd


RADAR_DIMENSIONS = [
    ("Scoring", "PTS_radar", "PTS_strength"),
    ("Rebounding", "REB_radar", "REB_strength"),
    ("Rim Protection", "BLK_radar", "BLK_strength"),
    ("Perimeter Defense", "STL_radar", "STL_strength"),
    ("Playmaking", "AST_radar", "AST_strength"),
    ("Three-Point Shooting", "FG3_PCT_radar", "FG3_PCT_strength"),
]


def player_radar_svg(row: pd.Series | dict[str, Any], size: int = 210) -> str:
    """Return an inline SVG radar chart for available player ability fields."""

    values = [_radar_value(row, radar_column, strength_column) for _, radar_column, strength_column in RADAR_DIMENSIONS]
    center = size / 2
    radius = size * 0.34
    label_radius = size * 0.44
    angles = [(-90 + index * 360 / len(RADAR_DIMENSIONS)) * math.pi / 180 for index in range(len(RADAR_DIMENSIONS))]

    grid_polygons = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        points = _points(center, radius * fraction, angles, [1.0] * len(RADAR_DIMENSIONS))
        grid_polygons.append(
            f'<polygon points="{points}" fill="none" stroke="#dfe4ea" stroke-width="1" />'
        )

    spokes = []
    labels = []
    for index, (label, _, _) in enumerate(RADAR_DIMENSIONS):
        angle = angles[index]
        x = center + radius * math.cos(angle)
        y = center + radius * math.sin(angle)
        label_x = center + label_radius * math.cos(angle)
        label_y = center + label_radius * math.sin(angle)
        anchor = "middle"
        if label_x < center - 8:
            anchor = "end"
        elif label_x > center + 8:
            anchor = "start"
        spokes.append(f'<line x1="{center:.1f}" y1="{center:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#edf0f4" stroke-width="1" />')
        labels.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" dominant-baseline="middle" font-size="9" fill="#5e6978">{escape(label)}</text>'
        )

    value_points = _points(center, radius, angles, [value / 100 for value in values])
    return f"""
    <div class="radar-wrap">
        <svg viewBox="0 0 {size} {size}" role="img" aria-label="Player ability radar chart">
            {''.join(grid_polygons)}
            {''.join(spokes)}
            <polygon points="{value_points}" fill="rgba(31,111,99,0.28)" stroke="#1f6f63" stroke-width="2" />
            {''.join(labels)}
        </svg>
    </div>
    """


def _radar_value(row: pd.Series | dict[str, Any], radar_column: str, strength_column: str) -> float:
    raw_value = _row_get(row, radar_column)
    if raw_value is None:
        raw_value = _row_get(row, strength_column)
        if raw_value is not None:
            raw_value = float(raw_value) * 25
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, value))


def _row_get(row: pd.Series | dict[str, Any], key: str) -> Any:
    if isinstance(row, pd.Series):
        value = row.get(key)
    else:
        value = row.get(key)
    if pd.isna(value):
        return None
    return value


def _points(center: float, radius: float, angles: list[float], values: list[float]) -> str:
    return " ".join(
        f"{center + radius * value * math.cos(angle):.1f},{center + radius * value * math.sin(angle):.1f}"
        for angle, value in zip(angles, values)
    )
