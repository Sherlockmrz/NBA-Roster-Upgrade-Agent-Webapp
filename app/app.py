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
    render_key_value_grid,
    render_parsed_query,
    render_recommendation_card,
    section_header,
    sidebar_values_from_state,
)
from nba_agent.agent import run_roster_agent
from nba_agent.llm.client import get_llm_status
from nba_agent.llm.parser import get_parser_warnings, parse_user_query
from nba_agent.visuals.charts import (
    fit_score_bar_chart,
    need_weight_bar_chart,
    need_weight_before_after_chart,
)


DATA_DIR = Path("data/raw")
EXPECTED_RAW_FILES = ["teams.csv", "games.csv", "games_details.csv"]
RANKING_MODES = ["Best Talent", "Realistic Fit", "Hidden Gems"]
EXAMPLE_QUERY = (
    "Recommend top 5 players for the Golden State Warriors to improve interior "
    "defense using the last 10 games. Only include players with at least 15 "
    "games and 15 average minutes."
)


st.set_page_config(
    page_title="NBA Roster Upgrade Agent",
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


def render_tool_a(need_df: pd.DataFrame) -> None:
    top_needs = need_df.sort_values("need_weight", ascending=False).head(3)
    columns = st.columns(3)
    for index, (_, row) in enumerate(top_needs.iterrows()):
        with columns[index]:
            st.metric(
                row["label"],
                f"{row['need_weight']:.2f}",
                help="Higher means this area is a larger deterministic roster need.",
            )
            if bool(row.get("goal_boosted", False)):
                st.caption("Goal boosted")

    need_weight_bar_chart(need_df)

    with st.expander("Full Tool A need table"):
        st.dataframe(need_df, use_container_width=True, hide_index=True)


def render_tool_b(player_strength_df: pd.DataFrame, filters: dict) -> None:
    st.metric("Candidate pool", f"{len(player_strength_df):,} players")
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
    st.info("Need reasoning completed with validated constraints.")
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

    st.dataframe(pd.DataFrame(multiplier_rows), use_container_width=True, hide_index=True)
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
    if ranked_df.empty:
        st.warning("No ranked players matched the current filters.")
        return

    for rank, (_, row) in enumerate(ranked_df.head(3).iterrows(), start=1):
        render_recommendation_card(rank, row)

    fit_score_bar_chart(ranked_df)

    with st.expander("Full Tool C ranked table"):
        st.dataframe(ranked_df, use_container_width=True, hide_index=True)


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
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def run_agent_from_state() -> None:
    query = st.session_state.get("user_query", "").strip()
    st.session_state["last_error"] = ""
    st.session_state["last_result"] = None
    st.session_state["last_debug_details"] = []
    st.session_state["last_parse_attempted"] = False

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

st.title("NBA Roster Upgrade Agent")
st.caption("Deterministic Streamlit interface for the Tool A / Tool B / Tool C pipeline.")

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
            st.caption("Deterministic fallback mode is active.")
    st.button(
        "Run Agent",
        type="primary",
        use_container_width=True,
        on_click=run_agent_from_state,
    )

if "salary" in user_query.lower():
    st.warning("Salary data is unavailable in the current dataset.")

if use_llm:
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

with st.container(border=True):
    section_header("Step 1: User Query", "The composed request passed to the agent.")
    st.write(result.user_query)

with st.container(border=True):
    section_header("Step 2: Parsed Query", "Validated parsing of team, goal, and filters.")
    render_parsed_query(result.parsed_query)

with st.container(border=True):
    section_header("Step 3: Agent Plan", "The ordered execution plan for the deterministic tools.")
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
        "Step 8: Final Scouting Summary",
        "Grounded deterministic summary. Salary, contracts, injuries, and rumors remain unavailable.",
    )
    st.write(result.final_summary)
