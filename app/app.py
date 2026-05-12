"""Streamlit entrypoint for the deterministic NBA roster-upgrade agent."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from components import (
    merge_parsed_with_sidebar,
    parsed_query_to_session_updates,
    render_help_box,
    render_hero,
    render_key_value_grid,
    render_parsed_query,
    render_recommendation_card,
    render_status_box,
    section_header,
    sidebar_values_from_state,
)
from nba_agent.agentic.tool_registry import (
    AGENTIC_TOOL_SELECTION,
    ALWAYS_VISIBLE_TOOLS,
    FINAL_SUMMARY,
    FULL_TOOL_PIPELINE,
    GROUNDED_QA,
    NEED_REASONING,
    PARSED_QUERY,
    SENSITIVITY,
    TOOL_A,
    TOOL_B,
    TOOL_C,
    USER_QUERY,
    ZERO_SHOT_EVALUATION,
    TOOL_REGISTRY,
)
from nba_agent.agentic.tool_selector import (
    ToolSelectionResult,
    deterministic_tool_selection,
    select_tools_for_query,
)
from nba_agent.agentic.summary import summary_for_selected_tools
from nba_agent.agent import run_roster_agent
from nba_agent.evaluation.metrics import (
    build_comparison_table,
    build_player_comparison_table,
    build_scored_candidate_table,
    evaluate_recommendations,
    tool_pipeline_explainability_checklist,
    tool_recommendations_from_ranked_df,
    zero_shot_explainability_checklist,
)
from nba_agent.evaluation.agent_benchmark import run_agent_tool_selection_benchmark
from nba_agent.llm.client import (
    clear_llm_call_history,
    get_llm_call_history,
    get_llm_status,
    test_llm_connection,
)
from nba_agent.llm.parser import get_parser_warnings, parse_user_query
from nba_agent.llm.qa import answer_grounded_question
from nba_agent.llm.zero_shot import ZeroShotResult, run_zero_shot_baseline
from nba_agent.visuals.charts import (
    fit_score_bar_chart,
    need_weight_bar_chart,
    need_weight_before_after_chart,
)
from nba_agent.visuals.radar import RADAR_DIMENSIONS, player_radar_svg


DATA_DIR = Path("data/raw")
EXPECTED_RAW_FILES = ["teams.csv", "games.csv", "games_details.csv"]
EXAMPLE_QUERY = (
    "Recommend the top 5 players for the Golden State Warriors to improve "
    "interior defense over the last 10 games. Only include players with at "
    "least 15 games and 15 average minutes. Check whether the ranking is "
    "robust, and keep a grounded Q&A section for follow-up questions."
)


st.set_page_config(
    page_title="NBA Roster Upgrade Agent WebApp",
    page_icon=":basketball:",
    layout="wide",
)


def load_css() -> None:
    css_path = Path(__file__).with_name("style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def missing_raw_files() -> list[str]:
    return [filename for filename in EXPECTED_RAW_FILES if not (DATA_DIR / filename).exists()]


@st.cache_data(show_spinner=False)
def load_teams_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "teams.csv", low_memory=False)


@st.cache_data(show_spinner=False)
def load_team_options() -> list[str]:
    teams = load_teams_df()
    labels = (
        teams["CITY"].fillna("").astype(str).str.strip()
        + " "
        + teams["NICKNAME"].fillna("").astype(str).str.strip()
    ).str.strip()
    labels = labels[labels != ""].sort_values().tolist()
    return labels or ["Warriors"]


def show_missing_data_error(missing: list[str]) -> None:
    st.error(
        "Required raw NBA data files are missing. The app expects these files under "
        "`data/raw/` before the deterministic agent can run."
    )
    st.code(
        "\n".join(
            [
                "data/raw/teams.csv",
                "data/raw/games.csv",
                "data/raw/games_details.csv",
            ]
        ),
        language="text",
    )
    st.caption(f"Currently missing: {', '.join(missing)}")


def compact_dataframe(df: pd.DataFrame, preferred_columns: list[str], rows: int = 5) -> pd.DataFrame:
    columns = [column for column in preferred_columns if column in df.columns]
    if not columns:
        return df.head(rows)
    return df[columns].head(rows)


def recommendation_preview_rows(result, limit: int = 5) -> list[pd.Series]:
    if result.ranked_df.empty:
        return []

    rows = []
    strength_df = result.player_strength_df
    for _, ranked_row in result.ranked_df.head(limit).iterrows():
        combined = ranked_row.to_dict()
        if not strength_df.empty and "PLAYER_NAME" in strength_df.columns:
            player_name = str(ranked_row.get("PLAYER_NAME", ""))
            match = strength_df[strength_df["PLAYER_NAME"].astype(str) == player_name]
            if "CURRENT_TEAM" in strength_df.columns and "CURRENT_TEAM" in ranked_row:
                current_team = str(ranked_row.get("CURRENT_TEAM", ""))
                team_match = match[match["CURRENT_TEAM"].astype(str) == current_team]
                if not team_match.empty:
                    match = team_match
            if not match.empty:
                combined = {**match.iloc[0].to_dict(), **combined}
        rows.append(pd.Series(combined))
    return rows


def player_profile_text(row: pd.Series) -> str:
    player_name = row.get("PLAYER_NAME", "This player")
    best_match = row.get("best_match", "the current adjusted needs")
    strongest = strongest_available_abilities(row)
    if strongest:
        strongest_text = ", ".join(strongest)
        return (
            f"{player_name} is recommended because his strongest available metrics "
            f"align with the team's adjusted needs, especially {best_match}. The "
            f"preview highlights {strongest_text} from the current dataset and Tool C fit scoring."
        )
    return (
        f"{player_name} is recommended because his strongest available metrics "
        f"align with the team's adjusted needs, especially {best_match}. This profile "
        "is based only on the current dataset and Tool C fit scoring."
    )


def strongest_available_abilities(row: pd.Series, limit: int = 2) -> list[str]:
    ability_values = []
    for label, radar_column, strength_column in RADAR_DIMENSIONS:
        value = row.get(radar_column, row.get(strength_column, 0.0))
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = 0.0
        ability_values.append((label, numeric_value))
    return [
        label
        for label, value in sorted(ability_values, key=lambda item: item[1], reverse=True)[:limit]
        if value > 0
    ]


def render_llm_control_status(llm_status, use_llm_requested: bool) -> None:
    mode = "On" if use_llm_requested else "Off"
    key_status = "available" if llm_status.available else "missing"
    runtime_mode = (
        f"Using LLM model: {llm_status.model}"
        if use_llm_requested and llm_status.available
        else "Deterministic fallback mode"
    )
    render_status_box(
        "Model status",
        f"LLM mode: {mode} · Model: {llm_status.model} · API key {key_status} · {runtime_mode}",
        tone="success" if use_llm_requested and llm_status.available else "neutral",
    )


def render_last_run_llm_status() -> None:
    requested = bool(st.session_state.get("last_llm_requested"))
    key_available = bool(st.session_state.get("last_llm_key_available"))
    fallback_used = bool(st.session_state.get("last_llm_fallback_used"))
    error_type = st.session_state.get("last_llm_error_type", "")
    model = st.session_state.get("last_llm_model") or get_llm_status().model

    if requested and key_available and error_type == "json_validation_error":
        message = "LLM response received but JSON validation failed · Deterministic fallback used"
        tone = "warning"
    elif requested and key_available and fallback_used:
        message = f"LLM requested but API call failed · Deterministic fallback used · Model attempted: {model}"
        tone = "warning"
    elif requested and key_available:
        message = f"LLM enabled · Model: {model}"
        tone = "success"
    elif requested:
        message = "LLM requested but API key missing · Deterministic fallback used"
        tone = "warning"
    else:
        message = "LLM disabled · Deterministic mode"
        tone = "neutral"
    render_status_box("Run mode", message, tone=tone)


def render_llm_debug_expander(debug_warnings: list[str]) -> None:
    with st.expander("LLM debug / fallback details"):
        st.caption(f"LLM enabled: {bool(st.session_state.get('last_llm_requested'))}")
        st.caption(f"API key available: {bool(st.session_state.get('last_llm_key_available'))}")
        st.caption(f"Model attempted: {st.session_state.get('last_llm_model') or get_llm_status().model}")
        st.caption(f"Fallback used: {bool(st.session_state.get('last_llm_fallback_used'))}")
        error_type = st.session_state.get("last_llm_error_type", "")
        error_message = st.session_state.get("last_llm_error_message", "")
        if error_type:
            st.caption(f"Error type: {error_type}")
        if error_message:
            st.caption(f"Error message: {error_message}")
        for detail in dict.fromkeys(debug_warnings):
            st.caption(detail)


def render_top_recommendations_preview(result) -> None:
    section_header(
        "Top 5 Recommended Players",
        "A quick front-office style view of the highest-ranked roster fits before the detailed reasoning trace.",
    )
    rows = recommendation_preview_rows(result, limit=5)
    if not rows:
        st.info("No recommendations available yet. Run the agent or adjust filters.")
        return

    for rank, row in enumerate(rows, start=1):
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="recommendation-card-head">
                    <span class="rank-badge">#{rank}</span>
                    <span class="fit-score-badge">Fit score {float(row.get("fit_score", 0.0)):.2f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            profile_col, radar_col = st.columns([2.25, 1.15])
            with profile_col:
                st.markdown(f"### {row.get('PLAYER_NAME', 'Unknown player')}")
                st.caption(f"Current team: {row.get('CURRENT_TEAM', 'Unknown team')}")
                st.markdown(f"**Best match:** {row.get('best_match', 'General fit')}")
                st.write(player_profile_text(row))
            with radar_col:
                st.markdown("**Ability radar**")
                st.markdown(player_radar_svg(row), unsafe_allow_html=True)


def render_summary_preview(result, summary=None) -> None:
    section_header(
        "AI Scouting Summary",
        "A concise preview of the grounded final summary before the full reasoning trace.",
    )
    with st.container(border=True):
        summary_text = summary.executive_summary if summary is not None else result.final_summary
        st.write(summary_text or result.scouting_summary.executive_summary)


def render_full_agent_pipeline(selection: ToolSelectionResult) -> None:
    section_header(
        "Full Agent Pipeline",
        "Select any pipeline card to open its output below without rerunning the agent.",
    )
    visible_tools = _visible_tools()
    manual_tools = _manual_tools()
    columns = st.columns(3)
    for index, definition in enumerate(FULL_TOOL_PIPELINE):
        with columns[index % 3]:
            status = _tool_card_status(definition.tool_id, selection, visible_tools, manual_tools)
            css_status = status.lower().replace(" ", "-").replace("/", "-")
            is_visible = definition.tool_id in visible_tools
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="tool-node-card {'tool-node-visible' if is_visible else 'tool-node-muted'}">
                        <div class="tool-node-topline">
                            <span class="tool-node-name">{definition.name}</span>
                            <span class="tool-status-badge status-{css_status}">{status}</span>
                        </div>
                        <p>{definition.description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                button_label = "Visible" if is_visible else "Open section"
                if st.button(
                    button_label,
                    key=f"tool_card_open_{definition.tool_id}",
                    use_container_width=True,
                ):
                    _open_tool_section(definition.tool_id)
                    st.rerun()


def render_tool_selection_decision(selection: ToolSelectionResult) -> None:
    section_header(
        "LLM Tool Selection Decision",
        "A validated planning layer decides which computed outputs should be shown first.",
    )
    source_label = "LLM" if selection.source == "llm" and not selection.used_fallback else "deterministic fallback"
    render_status_box(
        "Selection source",
        f"Tool selection came from {source_label}. {selection.validation_note}",
        tone="success" if selection.source == "llm" and not selection.used_fallback else "neutral",
    )

    selected_ids = list(selection.selected_tool_ids)
    skipped_ids = list(selection.skipped_tool_ids)
    selected_col, skipped_col = st.columns(2)
    with selected_col:
        st.markdown("**Selected tools**")
        if selected_ids:
            for tool_id in selected_ids:
                definition = TOOL_REGISTRY[tool_id]
                reason = selection.rationales.get(tool_id, "Selected for this query.")
                if tool_id in selection.required_dependency_ids:
                    reason = f"Required dependency. {reason}"
                st.markdown(f"- **{definition.name}**: {reason}")
        else:
            st.caption("No optional tools were selected.")
    with skipped_col:
        st.markdown("**Skipped tools**")
        if skipped_ids:
            for tool_id in skipped_ids:
                definition = TOOL_REGISTRY[tool_id]
                reason = selection.skipped_rationales.get(tool_id, "Not required by this query.")
                st.markdown(f"- **{definition.name}**: {reason}")
        else:
            st.caption("No optional tools were skipped.")

    if selection.warnings:
        with st.expander("Tool selection validation details"):
            for warning in selection.warnings:
                st.caption(warning)


def render_manual_unavailable_section(tool_id: str) -> None:
    definition = TOOL_REGISTRY[tool_id]
    with st.container(border=True):
        section_header(
            definition.name,
            "This pipeline node is available from the current app surface.",
        )
        if tool_id == ZERO_SHOT_EVALUATION:
            st.info("Open the Evaluation tab to run the zero-shot baseline comparison for this result.")
        else:
            st.info(
                "This tool was not run in the current execution. Run a query that selects it "
                "or enable it manually before execution."
            )


def _visible_tools() -> set[str]:
    return set(st.session_state.get("visible_tools", set(ALWAYS_VISIBLE_TOOLS)))


def _manual_tools() -> set[str]:
    return set(st.session_state.get("manual_visible_tools", set()))


def _open_tool_section(tool_id: str) -> None:
    visible = _visible_tools()
    manual = _manual_tools()
    visible.add(tool_id)
    if tool_id not in ALWAYS_VISIBLE_TOOLS:
        manual.add(tool_id)
    st.session_state["visible_tools"] = visible
    st.session_state["manual_visible_tools"] = manual


def _tool_card_status(
    tool_id: str,
    selection: ToolSelectionResult,
    visible_tools: set[str],
    manual_tools: set[str],
) -> str:
    if tool_id in ALWAYS_VISIBLE_TOOLS:
        return "Required dependency"
    if tool_id in manual_tools and tool_id in visible_tools and tool_id not in selection.selected_tool_ids:
        return "Manually opened"
    if tool_id in selection.required_dependency_ids:
        return "Required dependency"
    if tool_id in selection.selected_tool_ids:
        return "Selected by LLM" if selection.source == "llm" and not selection.used_fallback else "Selected by fallback"
    return "Available but not selected"


def render_evaluation_tab(result) -> None:
    top_k = int(result.parsed_query.top_k)
    zero_key = f"{result.user_query}|{top_k}|{st.session_state.get('selected_use_llm')}"

    section_header(
        "Evaluation Setup",
        "This page compares the zero-shot LLM answer and the tool pipeline answer under the same user query and user constraints.",
    )
    st.write(result.user_query)
    llm_status = get_llm_status()
    setup_cols = st.columns(4)
    setup_cols[0].metric("Top K", top_k)
    setup_cols[1].metric("Team", result.parsed_query.team_name)
    setup_cols[2].metric("Goal", result.parsed_query.goal or "not specified")
    setup_cols[3].metric("Zero-shot model", llm_status.model)
    render_key_value_grid(
        {
            "min_games": result.parsed_query.min_games,
            "min_avg_minutes": result.parsed_query.min_avg_minutes,
            "exclude_current_team": result.parsed_query.exclude_current_team,
            "zero_shot_llm_available": llm_status.available,
        }
    )
    st.caption(
        "We use the same query and evaluate both recommendation lists using the same "
        "dataset-grounded metrics. The zero-shot LLM can produce plausible names, "
        "but the tool pipeline can verify constraints, compute fit scores, measure "
        "need alignment, and test robustness."
    )

    section_header(
        "Zero-shot LLM Baseline",
        "The same user query is sent directly to the LLM without Tool A/B/C outputs.",
    )
    use_zero_shot_llm = bool(st.session_state.get("selected_use_llm"))
    if st.button("Run Zero-shot Baseline", use_container_width=True):
        st.session_state["zero_shot_result"] = run_zero_shot_baseline(
            user_query=result.user_query,
            top_k=top_k,
            use_llm=use_zero_shot_llm,
        )
        st.session_state["zero_shot_key"] = zero_key

    zero_result = st.session_state.get("zero_shot_result")
    if st.session_state.get("zero_shot_key") != zero_key:
        zero_result = run_zero_shot_baseline(
            user_query=result.user_query,
            top_k=top_k,
            use_llm=False,
        )

    render_zero_shot_result(zero_result)

    section_header(
        "Tool Pipeline Recommendations",
        "The current Tool C top-k output from the existing agent pipeline.",
    )
    tool_columns = [
        column
        for column in [
            "PLAYER_NAME",
            "CURRENT_TEAM",
            "GP",
            "AVG_MIN",
            "fit_score",
            "best_match",
        ]
        if column in result.ranked_df.columns
    ]
    tool_ranked = result.ranked_df.head(top_k).copy()
    if not tool_ranked.empty:
        tool_ranked.insert(0, "rank", range(1, len(tool_ranked) + 1))
        st.dataframe(tool_ranked[["rank", *tool_columns]], use_container_width=True, hide_index=True)
    else:
        st.info("No tool pipeline recommendations are available for the current filters.")

    section_header(
        "Fair Metric Evaluation",
        "Both recommendation lists are evaluated with the same dataset-grounded metrics.",
    )
    st.warning(
        "Important fairness note: Tool C fit score and need-alignment score are internal objective metrics. "
        "Our pipeline is designed to optimize them, so they are not independent ground-truth measures. "
        "We use them to show alignment with our explicit scoring objective, while fair audit metrics measure "
        "constraint satisfaction, dataset grounding, evidence coverage, and robustness."
    )
    render_help_box(
        "How to read this evaluation fairly",
        [
            "Zero-shot is evaluated using the same dataset checks after it produces names.",
            "Tool pipeline is expected to win on Tool C fit score because it optimizes Tool C.",
            "The stronger claim is not that our recommendations are always objectively better.",
            "The stronger claim is that our pipeline produces recommendations that are constraint-checked, dataset-grounded, scoreable, explainable, and robustness-tested.",
            "Zero-shot may produce plausible names, but it lacks built-in verification.",
        ],
    )
    zero_result = zero_result if isinstance(zero_result, ZeroShotResult) else run_zero_shot_baseline(
        result.user_query, top_k, use_llm=False
    )
    comparison_outputs = build_evaluation_outputs(result, zero_result)
    metric_cols = st.columns(3)
    metric_cols[0].metric(
        "Candidate found improvement",
        comparison_outputs["comparison_table"].iloc[0]["Improvement"],
    )
    metric_cols[1].metric(
        "Constraint improvement",
        comparison_outputs["comparison_table"].iloc[1]["Improvement"],
    )
    metric_cols[2].metric(
        "Evidence coverage improvement",
        comparison_outputs["comparison_table"].iloc[-1]["Improvement"],
    )

    section_header(
        "Main Comparison Table",
        "Metric-level comparison between zero-shot output and the tool pipeline.",
    )
    st.subheader("Fair Audit Metrics")
    st.caption(
        "These metrics compare both outputs under the same user query, constraints, and dataset checks."
    )
    fair_metrics = comparison_outputs["comparison_table"][
        comparison_outputs["comparison_table"]["Metric Type"] == "Fair audit metric"
    ]
    st.dataframe(
        fair_metrics,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Internal Objective Metrics")
    st.caption(
        "These metrics answer: how well does each recommendation list align with our explicit tool objective?"
    )
    objective_metrics = comparison_outputs["comparison_table"][
        comparison_outputs["comparison_table"]["Metric Type"] == "Internal objective metric"
    ]
    st.dataframe(
        objective_metrics,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Full combined comparison table"):
        st.dataframe(
            comparison_outputs["comparison_table"],
            use_container_width=True,
            hide_index=True,
        )

    section_header(
        "Player-by-player Comparison Table",
        "Side-by-side recommendation differences and dataset-grounded checks.",
    )
    st.dataframe(
        comparison_outputs["player_table"],
        use_container_width=True,
        hide_index=True,
    )

    section_header(
        "Explanation / Takeaway",
        "A concise interpretation of what the comparison is testing.",
    )
    render_status_box(
        "Evaluation takeaway",
        (
            "The zero-shot LLM baseline can produce plausible basketball recommendations, "
            "However, it does not automatically verify dataset constraints, compute fit "
            "scores, expose intermediate team-need reasoning, or test robustness. Our "
            "pipeline should not be interpreted as universally better just because it "
            "scores higher on Tool C; rather, it is more auditable because each "
            "recommendation is produced through explicit tools and can be evaluated step by step."
        ),
        tone="success",
    )

    render_agent_tool_selection_benchmark()


def render_zero_shot_result(zero_result) -> None:
    if not isinstance(zero_result, ZeroShotResult):
        st.info("Click Run Zero-shot Baseline to generate the LLM baseline.")
        return

    if not zero_result.players:
        st.info(zero_result.limitations)
    else:
        st.dataframe(pd.DataFrame(zero_result.players), use_container_width=True, hide_index=True)
        st.caption(zero_result.limitations)

    if zero_result.used_fallback or zero_result.error_type or zero_result.error_message:
        with st.expander("Zero-shot fallback / debug details"):
            st.caption(f"Model attempted: {zero_result.model}")
            st.caption(f"Fallback used: {zero_result.used_fallback}")
            if zero_result.error_type:
                st.caption(f"Error type: {zero_result.error_type}")
            if zero_result.error_message:
                st.caption(f"Error message: {zero_result.error_message}")


def build_evaluation_outputs(result, zero_result: ZeroShotResult) -> dict[str, pd.DataFrame]:
    need_df = (
        result.need_reasoning.adjusted_need_df
        if not result.need_reasoning.adjusted_need_df.empty
        else result.need_df
    )
    candidate_table = build_scored_candidate_table(need_df, result.player_strength_df)
    top_k = int(result.parsed_query.top_k)
    tool_recommendations = tool_recommendations_from_ranked_df(result.ranked_df, top_k)

    zero_metrics, zero_matches = evaluate_recommendations(
        zero_result.players,
        candidate_table,
        result.parsed_query,
        need_df,
        top_k,
        zero_shot_explainability_checklist(),
    )
    tool_metrics, tool_matches = evaluate_recommendations(
        tool_recommendations,
        candidate_table,
        result.parsed_query,
        need_df,
        top_k,
        tool_pipeline_explainability_checklist(result),
    )
    return {
        "comparison_table": build_comparison_table(
            zero_metrics, tool_metrics, result.sensitivity
        ),
        "player_table": build_player_comparison_table(
            zero_matches, tool_matches, result.parsed_query, top_k
        ),
    }


def render_agent_tool_selection_benchmark() -> None:
    section_header(
        "Agent Tool-Selection Benchmark",
        "A lightweight eval for whether the agent planner chooses appropriate tools for different query types.",
    )
    st.caption(
        "This benchmark treats tool selection as an eval target. It compares selected tools "
        "against expected tools, checks dependencies, and works in deterministic fallback mode "
        "when LLM access is unavailable."
    )
    benchmark_use_llm = bool(st.session_state.get("selected_use_llm"))
    if st.button("Run Agent Tool-Selection Benchmark", use_container_width=True):
        st.session_state["agent_tool_selection_benchmark"] = run_agent_tool_selection_benchmark(
            use_llm=benchmark_use_llm
        )

    benchmark = st.session_state.get("agent_tool_selection_benchmark")
    if benchmark is None:
        st.info("Click the benchmark button to evaluate tool-selection behavior.")
        return

    metrics = benchmark.aggregate_metrics
    metric_cols = st.columns(5)
    metric_cols[0].metric("Exact match", f"{metrics['average_exact_match']:.0%}")
    metric_cols[1].metric("Precision", f"{metrics['average_precision']:.0%}")
    metric_cols[2].metric("Recall", f"{metrics['average_recall']:.0%}")
    metric_cols[3].metric("F1", f"{metrics['average_f1']:.0%}")
    metric_cols[4].metric("Dependency valid", f"{metrics['dependency_validity_rate']:.0%}")
    st.dataframe(benchmark.to_dataframe(), use_container_width=True, hide_index=True)


def render_tool_a(need_df: pd.DataFrame) -> None:
    st.write(
        "Tool A turns recent team performance into need weights. Higher weights "
        "mean the model sees a larger roster weakness in that area."
    )
    top_needs = need_df.sort_values("need_weight", ascending=False).head(3)
    columns = st.columns(3)
    for index, (_, row) in enumerate(top_needs.iterrows()):
        with columns[index]:
            with st.container(border=True):
                st.metric(
                    row["label"],
                    f"{row['need_weight']:.2f}",
                    help="Higher means this area is a larger deterministic roster need.",
                )
                st.caption("Computed need weight")
            if bool(row.get("goal_boosted", False)):
                st.caption("Goal boosted")

    st.markdown("**Need weight chart**")
    need_weight_bar_chart(need_df)

    with st.expander("Full Tool A need table"):
        st.dataframe(need_df, use_container_width=True, hide_index=True)


def render_tool_b(player_strength_df: pd.DataFrame, filters: dict) -> None:
    st.write(
        "Tool B builds candidate strength profiles from the loaded box-score data "
        "after applying the parsed filters."
    )
    col_a, col_b = st.columns([1, 2])
    with col_a:
        with st.container(border=True):
            st.metric("Candidate pool", f"{len(player_strength_df):,} players")
            st.caption("Eligible players after filters")
    with col_b:
        render_key_value_grid(filters)

    st.markdown("**Top player strength rows**")
    st.dataframe(
        compact_dataframe(
            player_strength_df,
            [
                "PLAYER_NAME",
                "CURRENT_TEAM",
                "GP",
                "AVG_MIN",
                "REB_strength",
                "AST_strength",
                "STL_strength",
                "BLK_strength",
                "FG3_PCT_strength",
            ],
        ),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Full Tool B player strength table"):
        st.dataframe(player_strength_df, use_container_width=True, hide_index=True)


def render_need_reasoning(need_reasoning) -> None:
    st.write(
        "This step interprets the basketball goal and adjusts only the Tool A "
        "need weights. The LLM never calculates player stats or fit scores."
    )
    render_status_box(
        "Need reasoning",
        "Completed with validated constraints.",
        tone="success",
    )
    with st.container(border=True):
        st.markdown("**Tactical interpretation**")
        st.write(need_reasoning.tactical_interpretation)

    multiplier_rows = []
    for metric, multiplier in need_reasoning.metric_multipliers.items():
        label = metric
        adjusted = need_reasoning.adjusted_need_df
        if {"metric", "label"}.issubset(adjusted.columns):
            match = adjusted[adjusted["metric"] == metric]
            if not match.empty:
                label = match.iloc[0]["label"]
        multiplier_rows.append(
            {
                "metric": metric,
                "label": label,
                "multiplier": multiplier,
                "explanation": need_reasoning.explanations.get(metric, ""),
            }
        )

    st.markdown("**Validated metric multipliers**")
    st.dataframe(pd.DataFrame(multiplier_rows), use_container_width=True, hide_index=True)
    st.markdown("**Before / after need weights**")
    need_weight_before_after_chart(need_reasoning.adjusted_need_df)

    st.markdown("**Metric explanations**")
    explanation_columns = st.columns(2)
    for index, row in enumerate(multiplier_rows):
        with explanation_columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"**{row['label']}**")
                st.caption(f"Multiplier: {row['multiplier']:.2f}")
                st.write(row["explanation"])

    debug_lines = []
    if need_reasoning.used_fallback:
        debug_lines.append("Deterministic need-reasoning fallback was used.")
    debug_lines.extend(need_reasoning.warnings)
    if debug_lines:
        with st.expander("Need reasoning fallback / debug details"):
            for line in dict.fromkeys(debug_lines):
                st.caption(line)


def render_tool_c(ranked_df: pd.DataFrame) -> None:
    st.write(
        "Tool C matches adjusted team needs to player strength profiles. Scores "
        "are deterministic outputs from the current dataset."
    )
    if ranked_df.empty:
        st.warning("No ranked players matched the current filters.")
        return

    recommendation_columns = st.columns(3)
    for rank, (_, row) in enumerate(ranked_df.head(3).iterrows(), start=1):
        with recommendation_columns[(rank - 1) % 3]:
            render_recommendation_card(rank, row)

    st.markdown("**Fit score chart**")
    fit_score_bar_chart(ranked_df)

    with st.expander("Full Tool C ranked table"):
        st.dataframe(ranked_df, use_container_width=True, hide_index=True)


def render_sensitivity(sensitivity) -> None:
    st.write(
        "The robustness check perturbs adjusted need weights slightly and compares "
        "whether the same players remain near the top."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.metric("Stability", sensitivity.stability_label)
            st.caption("Higher overlap means the ranking is less fragile")
    with col_b:
        with st.container(border=True):
            st.metric("Top-k overlap", f"{sensitivity.top_k_overlap:.0%}")
            st.caption("Original top players retained after perturbation")
    st.write(sensitivity.explanation)

    with st.expander("Rank comparison table"):
        st.dataframe(
            sensitivity.rank_comparison_df,
            use_container_width=True,
            hide_index=True,
        )


def render_scouting_summary(summary) -> None:
    st.write(
        "The final summary converts the computed outputs into concise scouting "
        "language while keeping unsupported data unavailable."
    )
    with st.container(border=True):
        st.markdown("**Executive summary**")
        st.write(summary.executive_summary)

    st.markdown("**Key takeaways**")
    takeaway_columns = st.columns(3)
    for index, takeaway in enumerate(summary.key_takeaways[:3]):
        with takeaway_columns[index]:
            with st.container(border=True):
                st.markdown(f"**Takeaway {index + 1}**")
                st.write(takeaway)

    st.markdown("**Limitations**")
    st.caption(summary.limitations_note)

    debug_lines = []
    if summary.used_fallback:
        debug_lines.append("Deterministic scouting-summary fallback was used.")
    debug_lines.extend(summary.warnings)
    if debug_lines:
        with st.expander("Summary fallback / debug details"):
            for line in dict.fromkeys(debug_lines):
                st.caption(line)


def render_grounded_qa(result, use_llm: bool) -> None:
    st.caption(
        "Ask about the current run only. Answers are grounded in the displayed "
        "Tool A/B/C outputs, need reasoning, sensitivity, and summary."
    )
    example_questions = [
        "Why is the first player ranked first?",
        "What changed after LLM Need Reasoning?",
        "Why does this team need rebounding?",
        "Is the recommendation stable?",
        "What data is missing?",
    ]

    columns = st.columns(len(example_questions))
    for index, question in enumerate(example_questions):
        with columns[index]:
            if st.button(question, key=f"qa_example_{index}", use_container_width=True):
                _append_qa_exchange(result, question, use_llm)

    for message in st.session_state.get("qa_messages", []):
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a grounded question about this run")
    if question:
        _append_qa_exchange(result, question, use_llm)
        st.rerun()


def _append_qa_exchange(result, question: str, use_llm: bool) -> None:
    answer = answer_grounded_question(result, question, use_llm=use_llm)
    st.session_state.setdefault("qa_messages", [])
    st.session_state["qa_messages"].append({"role": "user", "content": question})
    st.session_state["qa_messages"].append({"role": "assistant", "content": answer})


def initialize_session_state(default_team: str) -> None:
    defaults = {
        "user_query": EXAMPLE_QUERY,
        "selected_team": default_team,
        "selected_goal": "interior defense",
        "selected_top_k": 5,
        "selected_recent_games": 10,
        "selected_min_games": 15,
        "selected_min_avg_minutes": 15.0,
        "selected_exclude_current_team": True,
        "selected_ranking_mode": "Best Talent",
        "selected_use_llm": False,
        "selected_use_sidebar_override": False,
        "last_result": None,
        "last_error": "",
        "last_debug_details": [],
        "last_parse_attempted": False,
        "last_llm_requested": False,
        "last_llm_key_available": False,
        "last_llm_model": get_llm_status().model,
        "last_llm_fallback_used": False,
        "last_llm_error_type": "",
        "last_llm_error_message": "",
        "llm_connection_test": None,
        "qa_messages": [],
        "last_tool_selection": None,
        "visible_tools": set(ALWAYS_VISIBLE_TOOLS),
        "manual_visible_tools": set(),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def run_agent_from_state() -> None:
    query = st.session_state.get("user_query", "").strip()
    st.session_state["last_error"] = ""
    st.session_state["last_result"] = None
    st.session_state["last_debug_details"] = []
    st.session_state["last_parse_attempted"] = False
    st.session_state["qa_messages"] = []
    clear_llm_call_history()

    if not query:
        st.session_state["last_error"] = "Enter a roster question before running the agent."
        return

    sidebar_values = sidebar_values_from_state(st.session_state)
    llm_status = get_llm_status()
    llm_requested = bool(st.session_state.get("selected_use_llm"))
    use_llm = bool(llm_requested and llm_status.available)
    st.session_state["last_llm_requested"] = llm_requested
    st.session_state["last_llm_key_available"] = llm_status.available
    st.session_state["last_llm_model"] = llm_status.model
    st.session_state["last_llm_fallback_used"] = bool(llm_requested and not llm_status.available)
    st.session_state["last_llm_error_type"] = "missing_api_key" if llm_requested and not llm_status.available else ""
    st.session_state["last_llm_error_message"] = "OpenRouter API key is missing." if llm_requested and not llm_status.available else ""
    if llm_requested and not llm_status.available:
        st.session_state["last_debug_details"].append(
            "Use LLM was enabled, but no OpenRouter key was available; deterministic parser was used."
        )

    parsed = parse_user_query(
        user_query=query,
        defaults=sidebar_values,
        teams_df=load_teams_df(),
        use_llm=use_llm,
    )
    st.session_state["last_parse_attempted"] = bool(use_llm)

    final_filters = merge_parsed_with_sidebar(
        parsed,
        sidebar_values,
        use_sidebar_override=bool(st.session_state.get("selected_use_sidebar_override")),
    )
    st.session_state.update(parsed_query_to_session_updates(final_filters))

    debug_details = []
    if use_llm:
        debug_details.append("LLM parsing attempted; invalid fields were normalized when needed.")
    debug_details.extend(get_parser_warnings())

    try:
        result = run_roster_agent(
            user_query=query,
            filters={**final_filters, "_parsed_fields": final_filters},
            data_dir="data/raw",
            use_llm=use_llm,
        )
    except Exception as exc:
        st.session_state["last_error"] = str(exc)
        return

    tool_selection = select_tools_for_query(
        user_query=query,
        parsed_query=result.parsed_query,
        use_llm=use_llm,
    )
    st.session_state["last_result"] = result
    st.session_state["last_tool_selection"] = tool_selection
    st.session_state["visible_tools"] = set(ALWAYS_VISIBLE_TOOLS) | set(tool_selection.selected_tool_ids)
    st.session_state["manual_visible_tools"] = set()
    st.session_state["last_debug_details"] = debug_details
    failed_call = next(
        (call for call in get_llm_call_history() if call.used_fallback or not call.ok),
        None,
    )
    if failed_call is not None:
        st.session_state["last_llm_fallback_used"] = True
        st.session_state["last_llm_error_type"] = failed_call.error_type
        st.session_state["last_llm_error_message"] = failed_call.error_message


load_css()

render_hero(
    "NBA Roster Upgrade Agent WebApp",
    (
        "An explainable LLM-powered front-office assistant for team diagnosis, "
        "player fit ranking, robustness checking, and grounded scouting Q&A."
    ),
)

missing = missing_raw_files()
if missing:
    show_missing_data_error(missing)
    st.stop()

team_options = load_team_options()
default_team_index = team_options.index("Golden State Warriors") if "Golden State Warriors" in team_options else 0
initialize_session_state(team_options[default_team_index])

user_query = st.text_area(
    "Ask a roster question",
    height=110,
    key="user_query",
)
llm_status = get_llm_status()
run_col, llm_col = st.columns([3, 1])
with run_col:
    st.button(
        "Run Agent",
        type="primary",
        use_container_width=True,
        on_click=run_agent_from_state,
        key="run_agent_main",
    )
with llm_col:
    st.toggle("Use LLM", key="selected_use_llm")
use_llm = bool(st.session_state.get("selected_use_llm"))
render_llm_control_status(llm_status, use_llm)
render_help_box(
    "How to read this result",
    [
        "Start with the parsed query to confirm the team, goal, and filters.",
        "Use Tool A and Need Reasoning to see why certain skills matter more.",
        "Read Tool C cards as deterministic fit recommendations from the current dataset.",
        "Use Sensitivity to judge whether the top recommendations are stable.",
        "Ask Grounded Q&A only about the displayed run outputs.",
    ],
)

with st.sidebar:
    st.header("Run Controls")
    if st.session_state["selected_team"] not in team_options:
        st.session_state["selected_team"] = team_options[default_team_index]
    team = st.selectbox("Team", team_options, key="selected_team")
    goal = st.text_input("Goal", key="selected_goal")
    top_k = st.slider("Top K", min_value=1, max_value=15, key="selected_top_k")
    recent_games = st.slider("Recent games", min_value=1, max_value=30, key="selected_recent_games")
    min_games = st.slider("Min games", min_value=1, max_value=82, key="selected_min_games")
    min_avg_minutes = st.slider(
        "Min average minutes",
        min_value=0.0,
        max_value=40.0,
        step=0.5,
        key="selected_min_avg_minutes",
    )
    exclude_current_team = st.checkbox("Exclude current team", key="selected_exclude_current_team")
    use_sidebar_override = st.checkbox("Use sidebar as manual override", key="selected_use_sidebar_override")
    llm_status = get_llm_status()
    with st.expander("LLM status", expanded=True):
        st.caption(f"LLM available: {'yes' if llm_status.available else 'no'}")
        st.caption(f"Model: {llm_status.model}")
        st.caption(f"API key: {'available' if llm_status.available else 'missing'}")
        if st.button("Test LLM connection", use_container_width=True):
            st.session_state["llm_connection_test"] = test_llm_connection().to_dict()
        test_result = st.session_state.get("llm_connection_test")
        if test_result:
            if test_result.get("ok"):
                st.success(f"Connection succeeded · Model: {test_result.get('model')}")
            else:
                st.warning("Connection failed.")
                if test_result.get("error_type"):
                    st.caption(f"Error type: {test_result.get('error_type')}")
                if test_result.get("error_message"):
                    st.caption(f"Error message: {test_result.get('error_message')}")
        if not llm_status.available:
            st.warning("OpenRouter API key is missing.")
            st.caption("Deterministic fallback mode is active.")

if not get_llm_status().available:
    render_status_box(
        "Fallback mode",
        "No API key is configured, so deterministic parsing and summaries remain available.",
        tone="warning",
    )
elif use_llm:
    st.info("LLM parsing and planning will be attempted, with deterministic fallback if unavailable.")

if st.session_state.get("last_error"):
    st.error("The deterministic agent could not complete this run.")
    st.caption(st.session_state["last_error"])
    st.stop()

result = st.session_state.get("last_result")
agent_tab, evaluation_tab = st.tabs(
    ["Agent Workflow", "Evaluation: Tool Pipeline vs Zero-shot"]
)
if result is None:
    with agent_tab:
        st.info("Type or edit a roster question, then click Run Agent. The sidebar follows the parsed query by default.")
    with evaluation_tab:
        st.info("Run the agent first to compare the tool pipeline against a zero-shot LLM baseline.")
    st.stop()

with agent_tab:
    tool_selection = st.session_state.get("last_tool_selection")
    if not isinstance(tool_selection, ToolSelectionResult):
        tool_selection = deterministic_tool_selection(result.user_query, result.parsed_query)
        st.session_state["last_tool_selection"] = tool_selection
    visible_tools = _visible_tools()

    render_last_run_llm_status()

    if st.session_state.get("last_parse_attempted"):
        st.info("LLM parsing attempted; validated query constraints are shown below.")

    debug_details = st.session_state.get("last_debug_details", [])
    quiet_warning_terms = ("LLM", "OpenRouter", "fallback", "parser", "planner")
    visible_warnings = []
    debug_warnings = list(debug_details)
    for warning in result.warnings:
        if any(term in warning for term in quiet_warning_terms):
            debug_warnings.append(warning)
        else:
            visible_warnings.append(warning)

    for warning in visible_warnings:
        st.warning(warning)

    render_llm_debug_expander(debug_warnings)

    render_full_agent_pipeline(tool_selection)
    render_tool_selection_decision(tool_selection)

    visible_tools = _visible_tools()
    display_summary = summary_for_selected_tools(result, visible_tools)
    if TOOL_C in visible_tools:
        render_top_recommendations_preview(result)
    if FINAL_SUMMARY in visible_tools:
        render_summary_preview(result, display_summary)

    if USER_QUERY in visible_tools:
        with st.container(border=True):
            section_header("Step 1: User Query", "What the user asked the agent to solve.")
            st.write(result.user_query)

    if PARSED_QUERY in visible_tools:
        with st.container(border=True):
            section_header("Step 2: Parsed Query", "Validated parsing of team, goal, and filters.")
            render_parsed_query(result.parsed_query)

    if AGENTIC_TOOL_SELECTION in visible_tools:
        with st.container(border=True):
            section_header(
                "Step 3: Agentic Tool Selection",
                "The fixed tool order is preserved while the display adapts to the query.",
            )
            st.markdown("**Validated display plan**")
            for item in result.agent_plan:
                if "ranking mode" in str(item).lower():
                    continue
                st.markdown(f"- {item}")
            st.markdown("**Visible tools for this run**")
            for tool_id in tool_selection.selected_tool_ids:
                st.markdown(f"- {TOOL_REGISTRY[tool_id].name}")

    if TOOL_A in visible_tools:
        with st.container(border=True):
            section_header(
                "Step 4: Tool A – Team Need Diagnosis",
                "Recent team weaknesses converted into need weights.",
            )
            render_tool_a(result.need_df)

    if NEED_REASONING in visible_tools:
        with st.container(border=True):
            section_header(
                "Step 5: LLM Need Reasoning",
                "Goal-aware need multipliers are validated before deterministic ranking.",
            )
            render_need_reasoning(result.need_reasoning)

    if TOOL_B in visible_tools:
        with st.container(border=True):
            section_header(
                "Step 6: Tool B – Player Strength Representation",
                "Candidate player vectors built from the loaded box-score dataset.",
            )
            render_tool_b(
                result.player_strength_df,
                {
                    "min_games": result.parsed_query.min_games,
                    "min_avg_minutes": result.parsed_query.min_avg_minutes,
                    "exclude_current_team": result.parsed_query.exclude_current_team,
                },
            )

    if TOOL_C in visible_tools:
        with st.container(border=True):
            section_header(
                "Step 7: Tool C – Fit Ranking",
                "Ranked matches between adjusted need weights and Tool B strengths.",
            )
            render_tool_c(result.ranked_df)

    if SENSITIVITY in visible_tools:
        with st.container(border=True):
            section_header(
                "Step 8: Sensitivity / Robustness Check",
                "Checks whether top recommendations hold under small need-weight changes.",
            )
            render_sensitivity(result.sensitivity)

    if FINAL_SUMMARY in visible_tools:
        with st.container(border=True):
            section_header(
                "Step 9: Final Scouting Summary",
                "Grounded deterministic summary based on the computed pipeline outputs.",
            )
            render_scouting_summary(display_summary)

    if GROUNDED_QA in visible_tools:
        with st.container(border=True):
            section_header(
                "Step 10: Grounded Q&A",
                "Ask follow-up questions that stay inside the current AgentResult.",
            )
            qa_use_llm = bool(st.session_state.get("selected_use_llm") and get_llm_status().available)
            render_grounded_qa(result, use_llm=qa_use_llm)

    if ZERO_SHOT_EVALUATION in visible_tools:
        render_manual_unavailable_section(ZERO_SHOT_EVALUATION)

with evaluation_tab:
    render_evaluation_tab(result)
