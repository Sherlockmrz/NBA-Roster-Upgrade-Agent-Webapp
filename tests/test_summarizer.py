from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_agent.llm.summarizer import summarize_scouting_report
from nba_agent.schemas import AnalysisRequest, NeedReasoningResult, SensitivityResult


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
        }
    )


def _need_reasoning() -> NeedReasoningResult:
    return NeedReasoningResult(
        tactical_interpretation="Interior defense emphasized rebounding and rim protection.",
        metric_multipliers={"REB": 1.35, "BLK": 1.5, "STL": 1.0},
        explanations={
            "REB": "Rebounding supports interior defense.",
            "BLK": "Rim protection supports interior defense.",
            "STL": "Perimeter defense remains baseline.",
        },
        adjusted_need_df=_need_df(),
        warnings=[],
        used_fallback=True,
    )


def _sensitivity() -> SensitivityResult:
    return SensitivityResult(
        stability_label="Stable",
        top_k_overlap=1.0,
        original_top_players=["Anthony Davis", "Rudy Gobert"],
        perturbed_top_players=["Anthony Davis", "Rudy Gobert"],
        explanation="Stable recommendation set.",
        rank_comparison_df=pd.DataFrame(),
    )


def test_fallback_summary_works_without_api_key():
    summary = summarize_scouting_report(
        AnalysisRequest(team_name="Golden State Warriors", goal="interior defense"),
        ["Run Tool A", "Run Tool B", "Run Tool C"],
        _need_df(),
        _need_df(),
        _need_reasoning(),
        _ranked_df(),
        _sensitivity(),
        use_llm=False,
    )

    assert summary.used_fallback is True
    assert summary.executive_summary
    assert len(summary.key_takeaways) == 3


def test_summary_mentions_salary_only_as_unavailable():
    summary = summarize_scouting_report(
        AnalysisRequest(team_name="Golden State Warriors", goal="interior defense"),
        [],
        _need_df(),
        _need_df(),
        _need_reasoning(),
        _ranked_df(),
        _sensitivity(),
        use_llm=False,
    )

    combined = " ".join(
        [summary.executive_summary, *summary.key_takeaways, summary.limitations_note]
    ).lower()
    assert "salary data" in combined
    assert "unavailable" in combined


def test_summary_references_players_needs_and_stability():
    summary = summarize_scouting_report(
        AnalysisRequest(team_name="Golden State Warriors", goal="interior defense"),
        [],
        _need_df(),
        _need_df(),
        _need_reasoning(),
        _ranked_df(),
        _sensitivity(),
        use_llm=False,
    )

    combined = " ".join([summary.executive_summary, *summary.key_takeaways]).lower()
    assert "anthony davis" in combined
    assert "rudy gobert" in combined
    assert "rebounding" in combined
    assert "stable" in combined
