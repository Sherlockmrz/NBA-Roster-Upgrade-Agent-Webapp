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
    render_workflow_strip,
    section_header,
    sidebar_values_from_state,
)
from nba_agent.agent import run_roster_agent
from nba_agent.llm.client import get_llm_status
from nba_agent.llm.parser import get_parser_warnings, parse_user_query
from nba_agent.llm.qa import answer_grounded_question
from nba_agent.visuals.charts import (
    fit_score_bar_chart,
    need_weight_bar_chart,
    need_weight_before_after_chart,
)
from nba_agent.visuals.radar import RADAR_DIMENSIONS, player_radar_svg


DATA_DIR = Path("data/raw")
EXPECTED_RAW_FILES = ["teams.csv", "games.csv", "games_details.csv"]
RANKING_MODES = ["Best Talent", "Realistic Fit", "Hidden Gems"]
EXAMPLE_QUERY = (
    "Recommend top 5 players for the Golden State Warriors to improve interior "
    "defense using the last 10 games. Only include players with at least 15 "
    "games and 15 average minutes."
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
            image_col, profile_col, radar_col = st.columns([1.05, 2.15, 1.25])
            with image_col:
                st.markdown(
                    '<div class="player-image-placeholder">Player image placeholder</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Salary: unavailable in current dataset")
            with profile_col:
                st.markdown(f"### {row.get('PLAYER_NAME', 'Unknown player')}")
                st.caption(f"Current team: {row.get('CURRENT_TEAM', 'Unknown team')}")
                st.markdown(f"**Best match:** {row.get('best_match', 'General fit')}")
                st.write(player_profile_text(row))
            with radar_col:
                st.markdown("**Ability radar**")
                st.markdown(player_radar_svg(row), unsafe_allow_html=True)


def render_summary_preview(result) -> None:
    section_header(
        "AI Scouting Summary",
        "A concise preview of the grounded final summary before the full reasoning trace.",
    )
    with st.container(border=True):
        st.write(result.final_summary or result.scouting_summary.executive_summary)


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
        "What changed after Feasibility Critique?",
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
        "qa_messages": [],
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

    if not query:
        st.session_state["last_error"] = "Enter a roster question before running the agent."
        return

    sidebar_values = sidebar_values_from_state(st.session_state)
    llm_status = get_llm_status()
    use_llm = bool(st.session_state.get("selected_use_llm") and llm_status.available)
    if st.session_state.get("selected_use_llm") and not llm_status.available:
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

    st.session_state["last_result"] = result
    st.session_state["last_debug_details"] = debug_details


load_css()

render_hero(
    "NBA Roster Upgrade Agent WebApp",
    (
        "An explainable LLM-powered front-office assistant for team diagnosis, "
        "player fit ranking, feasibility critique, and grounded scouting Q&A."
    ),
)
render_workflow_strip(
    ["Query", "Parse", "Diagnose", "Reason", "Rank", "Critique", "Verify", "Explain", "Chat"]
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
st.button(
    "Run Agent",
    type="primary",
    use_container_width=True,
    on_click=run_agent_from_state,
    key="run_agent_main",
)
render_help_box(
    "How to read this result",
    [
        "Start with the parsed query to confirm the team, goal, and filters.",
        "Use Tool A and Need Reasoning to see why certain skills matter more.",
        "Read Tool C cards as deterministic fit recommendations, not transaction feasibility.",
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
    ranking_mode = st.radio("Ranking mode", RANKING_MODES, key="selected_ranking_mode")
    use_sidebar_override = st.checkbox("Use sidebar as manual override", key="selected_use_sidebar_override")
    use_llm = st.toggle("Use LLM", key="selected_use_llm")
    llm_status = get_llm_status()
    with st.expander("LLM status", expanded=True):
        st.caption(f"LLM available: {'yes' if llm_status.available else 'no'}")
        st.caption(f"Model: {llm_status.model}")
        st.caption(f"Key preview: {llm_status.key_preview or 'not configured'}")
        if not llm_status.available:
            st.warning("OpenRouter API key is missing.")
            st.caption("Deterministic fallback mode is active.")

if "salary" in user_query.lower():
    st.warning("Salary data is unavailable in the current dataset.")

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
if result is None:
    st.info("Type or edit a roster question, then click Run Agent. The sidebar follows the parsed query by default.")
    st.stop()

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

if debug_warnings:
    with st.expander("LLM debug / fallback details"):
        for detail in dict.fromkeys(debug_warnings):
            st.caption(detail)

render_top_recommendations_preview(result)
render_summary_preview(result)

with st.container(border=True):
    section_header("Step 1: User Query", "What the user asked the agent to solve.")
    st.write(result.user_query)

with st.container(border=True):
    section_header("Step 2: Parsed Query", "Validated parsing of team, goal, and filters.")
    render_parsed_query(result.parsed_query)

with st.container(border=True):
    section_header("Step 3: Agent Plan", "The fixed workflow that keeps Tool A, Tool B, and Tool C in order.")
    for item in result.agent_plan:
        st.markdown(f"- {item}")

with st.container(border=True):
    section_header(
        "Step 4: Tool A – Team Need Diagnosis",
        "Recent team weaknesses converted into need weights.",
    )
    render_tool_a(result.need_df)

with st.container(border=True):
    section_header(
        "Step 5: LLM Need Reasoning",
        "Goal-aware need multipliers are validated before deterministic ranking.",
    )
    render_need_reasoning(result.need_reasoning)

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
            "ranking_mode": result.parsed_query.ranking_mode,
        },
    )

with st.container(border=True):
    section_header(
        "Step 7: Tool C – Fit Ranking",
        "Ranked matches between adjusted need weights and Tool B strengths.",
    )
    render_tool_c(result.ranked_df)

with st.container(border=True):
    section_header(
        "Step 8: Sensitivity / Robustness Check",
        "Checks whether top recommendations hold under small need-weight changes.",
    )
    render_sensitivity(result.sensitivity)

with st.container(border=True):
    section_header(
        "Step 9: Final Scouting Summary",
        "Grounded deterministic summary. Salary, contracts, injuries, and rumors remain unavailable.",
    )
    render_scouting_summary(result.scouting_summary)

with st.container(border=True):
    section_header(
        "Step 10: Grounded Q&A",
        "Ask follow-up questions that stay inside the current AgentResult.",
    )
    qa_use_llm = bool(st.session_state.get("selected_use_llm") and get_llm_status().available)
    render_grounded_qa(result, use_llm=qa_use_llm)
