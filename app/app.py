"""Streamlit entrypoint for the deterministic NBA roster-upgrade agent."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.components import (
    render_key_value_grid,
    render_parsed_query,
    render_recommendation_card,
    section_header,
)
from nba_agent.agent import run_roster_agent
from nba_agent.visuals.charts import fit_score_bar_chart, need_weight_bar_chart


DATA_DIR = Path("data/raw")
EXPECTED_RAW_FILES = ["teams.csv", "games.csv", "games_details.csv"]
RANKING_MODES = ["Best Talent", "Realistic Fit", "Hidden Gems"]


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
def load_team_options() -> list[str]:
    teams = pd.read_csv(DATA_DIR / "teams.csv", low_memory=False)
    labels = (
        teams["CITY"].fillna("").astype(str).str.strip()
        + " "
        + teams["NICKNAME"].fillna("").astype(str).str.strip()
    ).str.strip()
    labels = labels[labels != ""].sort_values().tolist()
    return labels or ["Warriors"]


def build_user_query(team: str, goal: str, top_k: int, recent_games: int) -> str:
    goal_text = goal.strip()
    if goal_text:
        return (
            f"Recommend top {top_k} players for the {team} to improve {goal_text} "
            f"using the last {recent_games} games."
        )
    return f"Recommend top {top_k} players for the {team} using the last {recent_games} games."


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


def render_tool_c(ranked_df: pd.DataFrame) -> None:
    if ranked_df.empty:
        st.warning("No ranked players matched the current filters.")
        return

    for rank, (_, row) in enumerate(ranked_df.head(3).iterrows(), start=1):
        render_recommendation_card(rank, row)

    fit_score_bar_chart(ranked_df)

    with st.expander("Full Tool C ranked table"):
        st.dataframe(ranked_df, use_container_width=True, hide_index=True)


load_css()

st.title("NBA Roster Upgrade Agent")
st.caption("Deterministic Streamlit interface for the Tool A / Tool B / Tool C pipeline.")

missing = missing_raw_files()
if missing:
    show_missing_data_error(missing)
    st.stop()

team_options = load_team_options()
default_team_index = team_options.index("Golden State Warriors") if "Golden State Warriors" in team_options else 0

with st.sidebar:
    st.header("Run Controls")
    team = st.selectbox("Team", team_options, index=default_team_index)
    goal = st.text_input("Goal", value="interior defense")
    top_k = st.slider("Top K", min_value=1, max_value=15, value=5)
    recent_games = st.slider("Recent games", min_value=1, max_value=30, value=10)
    min_games = st.slider("Min games", min_value=1, max_value=82, value=15)
    min_avg_minutes = st.slider(
        "Min average minutes",
        min_value=0.0,
        max_value=40.0,
        value=15.0,
        step=0.5,
    )
    exclude_current_team = st.checkbox("Exclude current team", value=True)
    ranking_mode = st.radio("Ranking mode", RANKING_MODES, index=0)
    use_llm = st.toggle("Use LLM", value=False)
    run_agent = st.button("Run Agent", type="primary", use_container_width=True)

filters = {
    "team": team,
    "goal": goal,
    "top_k": top_k,
    "recent_games": recent_games,
    "min_games": min_games,
    "min_avg_minutes": min_avg_minutes,
    "exclude_current_team": exclude_current_team,
    "ranking_mode": ranking_mode,
}
user_query = build_user_query(team, goal, top_k, recent_games)

if "salary" in user_query.lower():
    st.warning("Salary data is unavailable in the current dataset.")

if use_llm:
    st.info("LLM features are not implemented yet. This run will use deterministic fallback mode.")

if not run_agent:
    st.info("Choose controls in the sidebar, then run the deterministic agent.")
    st.stop()

try:
    with st.spinner("Running deterministic Tool A / Tool B / Tool C pipeline..."):
        result = run_roster_agent(
            user_query=user_query,
            filters=filters,
            data_dir="data/raw",
            use_llm=False,
        )
except FileNotFoundError as exc:
    show_missing_data_error(missing_raw_files() or EXPECTED_RAW_FILES)
    st.caption(str(exc))
    st.stop()
except Exception as exc:
    st.error("The deterministic agent could not complete this run.")
    st.caption(str(exc))
    st.stop()

for warning in result.warnings:
    st.warning(warning)

with st.container(border=True):
    section_header("Step 1: User Query", "The composed request passed to the agent.")
    st.write(result.user_query)

with st.container(border=True):
    section_header("Step 2: Parsed Query", "Deterministic parsing of team, goal, and filters.")
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
        "Step 5: Tool B – Player Strength Representation",
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
        "Step 6: Tool C – Fit Ranking",
        "Ranked matches between Tool A need weights and Tool B strengths.",
    )
    render_tool_c(result.ranked_df)

with st.container(border=True):
    section_header(
        "Step 7: Final Scouting Summary",
        "Grounded deterministic summary. Salary, contracts, injuries, and rumors remain unavailable.",
    )
    st.write(result.final_summary)
