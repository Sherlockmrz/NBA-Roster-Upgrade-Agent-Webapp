from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_agent.agent import run_roster_agent
from nba_agent.tools.sensitivity import run_sensitivity_check


def _need_df(adjusted: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "metric": ["REB", "AST", "STL", "BLK", "FG3_PCT"],
            "label": [
                "Rebounding",
                "Playmaking",
                "Perimeter Defense",
                "Rim Protection",
                "Three-Point Shooting",
            ],
            "need_weight": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    if adjusted:
        df["adjusted_need_weight"] = [10.0, 0.1, 0.1, 0.1, 0.1]
    return df


def _player_df() -> pd.DataFrame:
    return pd.DataFrame(
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
                "AST_strength": 2.0,
                "STL_strength": 0.0,
                "BLK_strength": 0.0,
                "FG3_PCT_strength": 0.0,
            },
        ]
    )


def test_sensitivity_returns_label_and_overlap():
    ranked = pd.DataFrame({"PLAYER_NAME": ["Rebounder", "Passer"]})

    result = run_sensitivity_check(_need_df(), _player_df(), ranked, top_k=2)

    assert result.stability_label in {"Stable", "Somewhat Stable", "Unstable"}
    assert isinstance(result.top_k_overlap, float)
    assert result.original_top_players == ["Rebounder", "Passer"]
    assert "PLAYER_NAME" in result.rank_comparison_df.columns


def test_sensitivity_does_not_crash_with_empty_inputs():
    result = run_sensitivity_check(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        top_k=5,
    )

    assert result.stability_label == "Unstable"
    assert result.top_k_overlap == 0.0
    assert result.perturbed_top_players == []


def test_sensitivity_uses_adjusted_need_weight_when_available():
    ranked = pd.DataFrame({"PLAYER_NAME": ["Rebounder", "Passer"]})

    result = run_sensitivity_check(_need_df(adjusted=True), _player_df(), ranked, top_k=2)

    assert result.perturbed_top_players[0] == "Rebounder"


def test_run_roster_agent_sensitivity_works_without_llm():
    result = run_roster_agent(
        "Recommend top 2 players for the Warriors to improve interior defense.",
        {"min_games": 15, "min_avg_minutes": 15},
        data_dir="data/raw",
        use_llm=False,
    )

    assert result.sensitivity.stability_label in {"Stable", "Somewhat Stable", "Unstable"}
    assert result.sensitivity.top_k_overlap >= 0.0
