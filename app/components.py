"""Reusable UI components for the Streamlit app."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import streamlit as st


def section_header(title: str, description: str = "") -> None:
    st.subheader(title)
    if description:
        st.caption(description)


def section_card(title: str, description: str = ""):
    """Return a Streamlit container styled as a workflow section."""

    container = st.container(border=True)
    with container:
        section_header(title, description)
    return container


def render_key_value_grid(values: dict[str, Any]) -> None:
    """Render compact key/value fields without dumping raw structures."""

    if not values:
        st.caption("No fields available.")
        return

    columns = st.columns(2)
    for index, (label, value) in enumerate(values.items()):
        with columns[index % 2]:
            st.markdown(f"**{label.replace('_', ' ').title()}**")
            st.caption(str(value))


def render_parsed_query(parsed_query: Any) -> None:
    """Render dataclass-like parsed query fields."""

    values = asdict(parsed_query) if hasattr(parsed_query, "__dataclass_fields__") else {}
    render_key_value_grid(values)


def render_recommendation_card(rank: int, row: Any) -> None:
    """Render one Tool C recommendation as a small card."""

    player_name = row.get("PLAYER_NAME", "Unknown player")
    current_team = row.get("CURRENT_TEAM", "Unknown team")
    fit_score = row.get("fit_score", 0)
    best_match = row.get("best_match", "General fit")
    games_played = row.get("GP", "n/a")
    avg_minutes = row.get("AVG_MIN", "n/a")

    with st.container(border=True):
        st.markdown(f"**#{rank} {player_name}**")
        st.caption(f"{current_team} | {best_match}")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Fit score", f"{float(fit_score):.2f}")
        col_b.metric("Games", f"{int(games_played)}" if games_played != "n/a" else "n/a")
        col_c.metric(
            "Avg min",
            f"{float(avg_minutes):.1f}" if avg_minutes != "n/a" else "n/a",
        )
