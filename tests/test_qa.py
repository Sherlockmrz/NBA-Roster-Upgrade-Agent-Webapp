from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_agent.llm.qa import SALARY_UNAVAILABLE, answer_grounded_question
from nba_agent.schemas import (
    AgentResult,
    AnalysisRequest,
    NeedReasoningResult,
    ScoutingSummaryResult,
    SensitivityResult,
)


def _need_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": ["REB", "BLK", "STL"],
            "label": ["Rebounding", "Rim Protection", "Perimeter Defense"],
            "need_weight": [1.2, 1.0, 0.8],
            "adjusted_need_weight": [1.6, 1.5, 0.8],
        }
    )


def _ranked_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PLAYER_NAME": ["Anthony Davis", "Rudy Gobert"],
            "CURRENT_TEAM": ["Los Angeles Lakers", "Minnesota Timberwolves"],
            "fit_score": [10.3, 7.6],
            "best_match": ["Rebounding + Rim Protection", "Rebounding + Rim Protection"],
            "GP": [55, 60],
            "AVG_MIN": [32.1, 30.4],
        }
    )


def _agent_result() -> AgentResult:
    need_df = _need_df()
    need_reasoning = NeedReasoningResult(
        tactical_interpretation="Interior defense emphasized rebounding and rim protection.",
        metric_multipliers={"REB": 1.35, "BLK": 1.5, "STL": 1.0},
        explanations={
            "REB": "Rebounding supports interior defense.",
            "BLK": "Rim protection supports interior defense.",
            "STL": "Perimeter defense remains baseline.",
        },
        adjusted_need_df=need_df,
        warnings=[],
        used_fallback=True,
    )
    sensitivity = SensitivityResult(
        stability_label="Stable",
        top_k_overlap=1.0,
        original_top_players=["Anthony Davis", "Rudy Gobert"],
        perturbed_top_players=["Anthony Davis", "Rudy Gobert"],
        explanation="Stable recommendation set.",
        rank_comparison_df=pd.DataFrame(),
    )
    summary = ScoutingSummaryResult(
        executive_summary="Golden State needs interior defense.",
        key_takeaways=[
            "Rebounding is a top need.",
            "Anthony Davis and Rudy Gobert fit.",
            "The recommendation is stable.",
        ],
        limitations_note="Salary data is unavailable in the current dataset.",
        used_fallback=True,
        warnings=[],
    )
    return AgentResult(
        user_query="Recommend top 5 players for the Warriors.",
        parsed_query=AnalysisRequest(
            team_name="Golden State Warriors",
            goal="interior defense",
            top_k=5,
        ),
        agent_plan=["Run Tool A", "Run Tool B", "Run Tool C"],
        need_df=need_df,
        need_reasoning=need_reasoning,
        player_strength_df=pd.DataFrame(),
        ranked_df=_ranked_df(),
        sensitivity=sensitivity,
        scouting_summary=summary,
        final_summary=summary.executive_summary,
        warnings=[],
        trace_steps=[],
    )


def test_salary_question_returns_unavailable_message():
    answer = answer_grounded_question(_agent_result(), "What is his salary?", use_llm=False)

    assert answer == SALARY_UNAVAILABLE


def test_first_player_question_uses_final_ranked_df():
    answer = answer_grounded_question(
        _agent_result(),
        "Why is this player ranked first?",
        use_llm=False,
    )

    assert "Anthony Davis" in answer
    assert "10.30" in answer


def test_stability_question_uses_sensitivity_output():
    answer = answer_grounded_question(
        _agent_result(),
        "Is this recommendation stable?",
        use_llm=False,
    )

    assert "Stable" in answer
    assert "100%" in answer


def test_no_api_key_still_gives_deterministic_fallback(monkeypatch):
    monkeypatch.setattr("nba_agent.llm.qa.is_llm_available", lambda: False)

    answer = answer_grounded_question(
        _agent_result(),
        "Why is the first player ranked first?",
        use_llm=True,
    )

    assert "Anthony Davis" in answer
