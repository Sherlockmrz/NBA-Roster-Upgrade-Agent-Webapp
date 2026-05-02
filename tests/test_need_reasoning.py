from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_agent.llm import reasoner
from nba_agent.llm.reasoner import reason_need_weights
from nba_agent.schemas import AnalysisRequest
from nba_agent.tools.fit_ranking import rank_players_by_fit
from nba_agent.tools.need_diagnosis import CORE_METRIC_LABELS, NEED_COLUMNS


def _need_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": ["REB", "AST", "STL", "BLK", "FG3_PCT"],
            "label": [
                "Rebounding",
                "Playmaking",
                "Perimeter Defense",
                "Rim Protection",
                "Three-Point Shooting",
            ],
            "team_value": [40, 20, 6, 4, 0.33],
            "league_mean": [42, 24, 7, 5, 0.36],
            "z_score": [-0.5, -0.4, -0.2, -0.6, -0.3],
            "need_weight": [1.0, 1.0, 1.0, 1.0, 1.0],
            "goal_boosted": [False, False, False, False, False],
        }
    )[NEED_COLUMNS]


def test_interior_defense_boosts_reb_and_blk_in_fallback_mode():
    result = reason_need_weights(
        AnalysisRequest(team_name="Golden State Warriors", goal="interior defense"),
        _need_df(),
        CORE_METRIC_LABELS,
        use_llm=False,
    )

    assert result.used_fallback is True
    assert result.metric_multipliers["REB"] > 1.0
    assert result.metric_multipliers["BLK"] > 1.0
    assert result.metric_multipliers["AST"] == 1.0


def test_llm_multipliers_are_bounded_and_invalid_metrics_ignored(monkeypatch):
    def fake_call_llm_json(messages, fallback):
        return {
            "tactical_interpretation": "Emphasize glass and creation.",
            "metric_multipliers": {
                "REB": 9,
                "AST": 0.1,
                "NOT_A_METRIC": 2,
            },
            "explanations": {
                "REB": "Rebounding supports the goal.",
                "NOT_A_METRIC": "Should be ignored.",
            },
        }

    monkeypatch.setattr(reasoner, "call_llm_json", fake_call_llm_json)

    result = reason_need_weights(
        AnalysisRequest(team_name="Golden State Warriors", goal="rebounding"),
        _need_df(),
        CORE_METRIC_LABELS,
        use_llm=True,
    )

    assert result.used_fallback is False
    assert result.metric_multipliers["REB"] == 2.0
    assert result.metric_multipliers["AST"] == 0.5
    assert "NOT_A_METRIC" not in result.metric_multipliers
    assert any("NOT_A_METRIC" in warning for warning in result.warnings)


def test_adjusted_need_df_keeps_original_columns_plus_adjusted_weight():
    need_df = _need_df()

    result = reason_need_weights(
        AnalysisRequest(team_name="Golden State Warriors", goal="shooting"),
        need_df,
        CORE_METRIC_LABELS,
        use_llm=False,
    )

    assert list(result.adjusted_need_df.columns) == list(need_df.columns) + [
        "adjusted_need_weight"
    ]
    assert "adjusted_need_weight" not in need_df.columns


def test_tool_c_uses_adjusted_need_weight_when_available():
    need_df = _need_df()
    adjusted_need_df = need_df.copy()
    adjusted_need_df["adjusted_need_weight"] = [10.0, 0.1, 0.1, 0.1, 0.1]

    player_df = pd.DataFrame(
        [
            {
                "PLAYER_NAME": "Rebounder",
                "CURRENT_TEAM": "A",
                "TEAM_ID": 1,
                "GP": 20,
                "AVG_MIN": 20,
                "REB_strength": 2.0,
                "AST_strength": 0.0,
                "STL_strength": 0.0,
                "BLK_strength": 0.0,
                "FG3_PCT_strength": 0.0,
            },
            {
                "PLAYER_NAME": "Passer",
                "CURRENT_TEAM": "B",
                "TEAM_ID": 2,
                "GP": 20,
                "AVG_MIN": 20,
                "REB_strength": 0.0,
                "AST_strength": 9.0,
                "STL_strength": 0.0,
                "BLK_strength": 0.0,
                "FG3_PCT_strength": 0.0,
            },
        ]
    )

    ranked = rank_players_by_fit(adjusted_need_df, player_df, top_k=2, exclude_current_team=False)

    assert ranked.iloc[0]["PLAYER_NAME"] == "Rebounder"
