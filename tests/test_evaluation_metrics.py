from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_agent.evaluation.metrics import (
    RecommendationMetrics,
    build_comparison_table,
    compute_average_fit_score,
    compute_candidate_found_rate,
    compute_constraint_satisfaction_rate,
    compute_explainability_score,
    compute_need_alignment_score,
    match_recommended_players,
    normalize_player_name,
)
from nba_agent.llm.zero_shot import run_zero_shot_baseline
from nba_agent.schemas import AnalysisRequest, SensitivityResult


def _candidate_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PLAYER_NAME": ["Anthony Davis", "Rudy Gobert"],
            "CURRENT_TEAM": ["Los Angeles Lakers", "Minnesota Timberwolves"],
            "GP": [28, 10],
            "AVG_MIN": [32.0, 12.0],
            "fit_score": [10.0, 4.0],
            "REB_strength": [1.5, 1.4],
            "BLK_strength": [1.8, 1.7],
        }
    )


def _need_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": ["REB", "BLK"],
            "adjusted_need_weight": [1.5, 1.4],
        }
    )


def _query() -> AnalysisRequest:
    return AnalysisRequest(
        team_name="Golden State Warriors",
        min_games=15,
        min_avg_minutes=15,
        exclude_current_team=True,
    )


def test_normalize_player_name_removes_punctuation_and_spaces():
    assert normalize_player_name(" Anthony   Davis, Jr. ") == "anthony davis jr"


def test_unmatched_zero_shot_players_count_as_not_found():
    matches = match_recommended_players(
        [{"rank": 1, "player_name": "Imaginary Player"}],
        _candidate_table(),
    )

    assert matches[0]["found"] is False
    assert compute_candidate_found_rate(matches, top_k=1) == 0.0


def test_constraint_satisfaction_counts_unmatched_as_failing():
    matches = match_recommended_players(
        [{"rank": 1, "player_name": "Imaginary Player"}],
        _candidate_table(),
    )

    assert compute_constraint_satisfaction_rate(matches, _query(), top_k=1) == 0.0


def test_average_fit_score_handles_missing_and_penalized_unmatched():
    matches = match_recommended_players(
        [
            {"rank": 1, "player_name": "Anthony Davis"},
            {"rank": 2, "player_name": "Imaginary Player"},
        ],
        _candidate_table(),
    )

    assert compute_average_fit_score(matches, top_k=2, penalize_unmatched=False) == 10.0
    assert compute_average_fit_score(matches, top_k=2, penalize_unmatched=True) == 5.0


def test_need_alignment_returns_none_when_strength_columns_missing():
    candidate = _candidate_table().drop(columns=["REB_strength", "BLK_strength"])
    matches = match_recommended_players(
        [{"rank": 1, "player_name": "Anthony Davis"}],
        candidate,
    )

    assert compute_need_alignment_score(matches, _need_df(), top_k=1) is None


def test_improvement_calculation_avoids_division_by_zero():
    zero = RecommendationMetrics(0, 0, 0.0, 0.0, 0.0, 0)
    tool = RecommendationMetrics(1, 1, 3.0, 3.0, 2.0, 1)
    sensitivity = SensitivityResult(
        stability_label="Stable",
        top_k_overlap=1.0,
        original_top_players=[],
        perturbed_top_players=[],
        explanation="",
        rank_comparison_df=pd.DataFrame(),
    )

    table = build_comparison_table(zero, tool, sensitivity)

    assert "relative N/A" in table.loc[2, "Improvement"]
    assert "Metric Type" in table.columns
    assert table.loc[2, "Metric Type"] == "Internal objective metric"
    assert "objective-alignment difference" in table.loc[2, "Improvement"]


def test_comparison_table_labels_fair_audit_metrics():
    zero = RecommendationMetrics(0, 0, 0.0, 0.0, 0.0, 0)
    tool = RecommendationMetrics(1, 1, 3.0, 3.0, 2.0, 1)
    sensitivity = SensitivityResult(
        stability_label="Stable",
        top_k_overlap=1.0,
        original_top_players=[],
        perturbed_top_players=[],
        explanation="",
        rank_comparison_df=pd.DataFrame(),
    )

    table = build_comparison_table(zero, tool, sensitivity)

    fair_rows = table[table["Metric Type"] == "Fair audit metric"]
    assert "Candidate Found Rate" in fair_rows["Metric"].tolist()
    assert "Constraint Satisfaction Rate" in fair_rows["Metric"].tolist()
    assert "Evidence Coverage / Explainability Score" in fair_rows["Metric"].tolist()
    assert all("audit improvement" in value for value in fair_rows["Improvement"])


def test_evidence_coverage_score_computes_correctly():
    checklist = {"a": True, "b": False, "c": True, "d": False}

    assert compute_explainability_score(checklist) == 0.5


def test_zero_shot_unavailable_when_llm_disabled():
    result = run_zero_shot_baseline("Recommend players.", top_k=5, use_llm=False)

    assert result.players == []
    assert result.used_fallback is True
    assert "requires LLM access" in result.limitations
