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

    chart_df = adjusted_need_df.copy()
    chart_df["label"] = chart_df["label"].fillna(chart_df.get("metric", "Unknown metric"))
    category_order = chart_df["label"].astype(str).tolist()
    chart_df["need_weight"] = pd.to_numeric(chart_df["need_weight"], errors="coerce").fillna(0.0)
    chart_df["adjusted_need_weight"] = (
        pd.to_numeric(chart_df["adjusted_need_weight"], errors="coerce").fillna(0.0)
    )
    chart_df = chart_df[["label", "need_weight", "adjusted_need_weight"]].rename(
        columns={
            "need_weight": "Original need weight",
            "adjusted_need_weight": "Adjusted need weight",
        }
    )
    long_df = chart_df.melt(
        id_vars="label",
        var_name="series",
        value_name="value",
    )
    long_df["label"] = pd.Categorical(
        long_df["label"].astype(str),
        categories=category_order,
        ordered=True,
    )
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce").fillna(0.0)
    long_df["value_label"] = long_df["value"].map(lambda value: f"{value:.2f}")

    st.vega_lite_chart(
        long_df,
        {
            "height": max(260, 46 * len(category_order)),
            "layer": [
                {
                    "mark": {"type": "bar", "tooltip": True},
                    "encoding": {
                        "x": {
                            "field": "value",
                            "type": "quantitative",
                            "title": "Weight",
                        },
                        "y": {
                            "field": "label",
                            "type": "nominal",
                            "title": None,
                            "sort": category_order,
                        },
                        "yOffset": {"field": "series"},
                        "color": {
                            "field": "series",
                            "type": "nominal",
                            "title": None,
                        },
                    },
                },
                {
                    "mark": {"type": "text", "align": "left", "baseline": "middle", "dx": 4},
                    "encoding": {
                        "x": {"field": "value", "type": "quantitative"},
                        "y": {
                            "field": "label",
                            "type": "nominal",
                            "sort": category_order,
                        },
                        "yOffset": {"field": "series"},
                        "text": {"field": "value_label"},
                        "color": {"value": "#18202b"},
                    },
                },
            ],
            "resolve": {"scale": {"y": "shared"}},
        },
        use_container_width=True,
    )
