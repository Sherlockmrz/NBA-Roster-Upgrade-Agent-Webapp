from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_agent.agentic.summary import summary_for_selected_tools
from nba_agent.agentic.tool_registry import FINAL_SUMMARY, NEED_REASONING, TOOL_A, TOOL_C
from nba_agent.schemas import (
    AgentResult,
    AnalysisRequest,
    NeedReasoningResult,
    ScoutingSummaryResult,
    SensitivityResult,
)


def _agent_result() -> AgentResult:
    need_df = pd.DataFrame(
        {
            "metric": ["REB", "BLK"],
            "label": ["Rebounding", "Rim Protection"],
            "need_weight": [1.2, 1.1],
            "adjusted_need_weight": [1.6, 1.5],
        }
    )
    need_reasoning = NeedReasoningResult(
        tactical_interpretation="Interior defense emphasizes rebounding and rim protection.",
        metric_multipliers={"REB": 1.3, "BLK": 1.4},
        explanations={},
        adjusted_need_df=need_df,
        warnings=[],
        used_fallback=True,
    )
    summary = ScoutingSummaryResult(
        executive_summary="Top recommended players have strong fit scores.",
        key_takeaways=[
            "Top players match the need profile.",
            "Fit scores support the ranking.",
            "Ranked players are stable.",
        ],
        limitations_note="Dataset limitations.",
        used_fallback=True,
        warnings=[],
    )
    return AgentResult(
        user_query="Explain which team needs matter most.",
        parsed_query=AnalysisRequest(
            team_name="Golden State Warriors",
            goal="interior defense",
        ),
        agent_plan=[],
        need_df=need_df,
        need_reasoning=need_reasoning,
        player_strength_df=pd.DataFrame(),
        ranked_df=pd.DataFrame(
            {
                "PLAYER_NAME": ["Example Player"],
                "fit_score": [1.0],
                "best_match": ["Rebounding"],
            }
        ),
        sensitivity=SensitivityResult(
            stability_label="Stable",
            top_k_overlap=1.0,
            original_top_players=[],
            perturbed_top_players=[],
            explanation="Stable.",
            rank_comparison_df=pd.DataFrame(),
        ),
        scouting_summary=summary,
        final_summary=summary.executive_summary,
        warnings=[],
        trace_steps=[],
    )


def test_need_only_summary_does_not_mention_player_ranking_outputs():
    summary = summary_for_selected_tools(
        _agent_result(),
        {TOOL_A, NEED_REASONING, FINAL_SUMMARY},
    )

    combined = " ".join(
        [summary.executive_summary, *summary.key_takeaways, summary.limitations_note]
    ).lower()
    assert "player recommendations" not in combined
    assert "top players" not in combined
    assert "fit scores" not in combined
    assert "ranked players" not in combined
    assert "rebounding" in combined
    assert "rim protection" in combined


def test_tool_c_selected_uses_existing_pipeline_summary():
    result = _agent_result()
    summary = summary_for_selected_tools(result, {TOOL_C, FINAL_SUMMARY})

    assert summary is result.scouting_summary
