"""Small Streamlit chart helpers for deterministic roster outputs."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def need_weight_bar_chart(need_df: pd.DataFrame) -> None:
    """Render Tool A need weights as a compact horizontal bar chart."""

    if need_df.empty or not {"label", "need_weight"}.issubset(need_df.columns):
        st.info("No need weights are available to chart.")
        return

    chart_df = (
        need_df[["label", "need_weight"]]
        .sort_values("need_weight", ascending=True)
        .set_index("label")
    )
    st.bar_chart(chart_df)


def fit_score_bar_chart(ranked_df: pd.DataFrame) -> None:
    """Render Tool C fit scores by player."""

    if ranked_df.empty or not {"PLAYER_NAME", "fit_score"}.issubset(ranked_df.columns):
        st.info("No fit scores are available to chart.")
        return

    chart_df = (
        ranked_df[["PLAYER_NAME", "fit_score"]]
        .sort_values("fit_score", ascending=True)
        .set_index("PLAYER_NAME")
    )
    st.bar_chart(chart_df)


def need_weight_before_after_chart(adjusted_need_df: pd.DataFrame) -> None:
    """Render original and adjusted need weights side by side."""

    required = {"label", "need_weight", "adjusted_need_weight"}
    if adjusted_need_df.empty or not required.issubset(adjusted_need_df.columns):
        st.info("No adjusted need weights are available to chart.")
        return

    chart_df = (
        adjusted_need_df[["label", "need_weight", "adjusted_need_weight"]]
        .sort_values("adjusted_need_weight", ascending=True)
        .set_index("label")
    )
    st.bar_chart(chart_df)
