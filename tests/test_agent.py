from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_agent.agent import run_roster_agent
from nba_agent.schemas import AgentResult


def test_run_roster_agent_deterministic_trace_and_outputs():
    result = run_roster_agent(
        "Recommend top 3 players for the Warriors to improve interior defense using salary constraints",
        {"min_games": 20, "min_avg_minutes": 18},
        data_dir="data/raw",
        use_llm=False,
    )

    assert isinstance(result, AgentResult)
    assert "Salary data is unavailable in the current dataset." in result.warnings
    assert [step.title for step in result.trace_steps] == [
        "User Query",
        "Parsed Query",
        "Agent Plan",
        "Tool A: Team Need Diagnosis",
        "LLM Need Reasoning",
        "Tool B: Player Strength Representation",
        "Tool C: Fit Ranking",
        "Final Scouting Summary",
    ]
    assert [step.step_number for step in result.trace_steps] == list(range(1, 9))
    assert result.parsed_query.team_name == "Golden State Warriors"
    assert len(result.ranked_df) == 3
    assert not result.need_df.empty
    assert not result.need_reasoning.adjusted_need_df.empty
    assert not result.player_strength_df.empty
    assert "salary" in result.final_summary.lower()


def test_run_roster_agent_with_complete_filters():
    result = run_roster_agent(
        "Recommend top 4 players for the Warriors to improve interior defense.",
        {
            "team": "Warriors",
            "goal": "interior defense",
            "top_k": 4,
            "recent_games": 10,
            "min_games": 15,
            "min_avg_minutes": 15,
            "exclude_current_team": True,
            "ranking_mode": "Best Talent",
        },
        data_dir="data/raw",
        use_llm=False,
    )

    assert result.warnings == []
    assert result.parsed_query.team_name == "Golden State Warriors"
    assert result.parsed_query.goal == "interior defense"
    assert result.parsed_query.ranking_mode == "Best Talent"
    assert len(result.ranked_df) == 4


def test_run_roster_agent_with_missing_optional_filters_uses_defaults():
    result = run_roster_agent(
        "Recommend players for the Warriors.",
        {"team": "Warriors"},
        data_dir="data/raw",
        use_llm=False,
    )

    assert result.parsed_query.top_k == 5
    assert result.parsed_query.recent_games == 10
    assert result.parsed_query.min_games == 15
    assert result.parsed_query.min_avg_minutes == 15
    assert result.parsed_query.exclude_current_team is True
    assert result.parsed_query.ranking_mode == "Best Talent"
    assert any("top_k missing; defaulted to 5." == warning for warning in result.warnings)
    assert len(result.ranked_df) == 5


def test_run_roster_agent_infers_missing_team_and_goal_from_clear_query():
    result = run_roster_agent(
        "Recommend top 2 players for the Lakers to improve rebounding.",
        {"min_games": 15, "min_avg_minutes": 15},
        data_dir="data/raw",
        use_llm=False,
    )

    assert result.parsed_query.team_name == "Los Angeles Lakers"
    assert result.parsed_query.goal == "rebounding"
    assert result.parsed_query.top_k == 2
    assert any("Team filter missing; inferred team from query: Los Angeles Lakers." == warning for warning in result.warnings)
    assert any("Goal filter missing; inferred goal from query: rebounding." == warning for warning in result.warnings)
    assert len(result.ranked_df) == 2
