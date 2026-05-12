import nba_agent.agentic.tool_selector as selector_module
from nba_agent.agentic.tool_registry import (
    FINAL_SUMMARY,
    GROUNDED_QA,
    NEED_REASONING,
    SENSITIVITY,
    TOOL_A,
    TOOL_B,
    TOOL_C,
)
from nba_agent.agentic.tool_selector import (
    deterministic_tool_selection,
    select_tools_for_query,
    validate_tool_selection,
)
from nba_agent.llm.client import LLMCallResult


DEFAULT_QUERY = (
    "Recommend the top 5 players for the Golden State Warriors to improve "
    "interior defense over the last 10 games. Only include players with at "
    "least 15 games and 15 average minutes. Check whether the ranking is "
    "robust, and keep a grounded Q&A section for follow-up questions."
)


def test_default_query_selects_expected_agent_tools():
    result = deterministic_tool_selection(DEFAULT_QUERY)

    assert TOOL_A in result.selected_tool_ids
    assert NEED_REASONING in result.selected_tool_ids
    assert TOOL_B in result.selected_tool_ids
    assert TOOL_C in result.selected_tool_ids
    assert SENSITIVITY in result.selected_tool_ids
    assert FINAL_SUMMARY in result.selected_tool_ids
    assert GROUNDED_QA in result.selected_tool_ids


def test_team_need_explanation_query_skips_player_ranking_tools():
    result = deterministic_tool_selection(
        "For the Golden State Warriors, explain which team needs matter most "
        "if the goal is improving interior defense."
    )

    assert result.selected_tool_ids == (TOOL_A, NEED_REASONING, FINAL_SUMMARY)
    assert TOOL_B not in result.selected_tool_ids
    assert TOOL_C not in result.selected_tool_ids
    assert SENSITIVITY not in result.selected_tool_ids
    assert GROUNDED_QA not in result.selected_tool_ids


def test_recommendation_query_selects_core_ranking_tools():
    result = deterministic_tool_selection(
        "Recommend the top 5 players for the Golden State Warriors to improve interior defense."
    )

    assert result.selected_tool_ids == (TOOL_A, NEED_REASONING, TOOL_B, TOOL_C, FINAL_SUMMARY)


def test_stability_query_includes_sensitivity_check():
    result = deterministic_tool_selection(
        "Recommend the top 5 players and check whether the ranking is robust."
    )

    assert TOOL_A in result.selected_tool_ids
    assert NEED_REASONING in result.selected_tool_ids
    assert TOOL_B in result.selected_tool_ids
    assert TOOL_C in result.selected_tool_ids
    assert SENSITIVITY in result.selected_tool_ids
    assert FINAL_SUMMARY in result.selected_tool_ids


def test_follow_up_query_includes_grounded_qa():
    result = deterministic_tool_selection(
        "Recommend players for the Knicks and keep a chat for follow-up questions."
    )

    assert GROUNDED_QA in result.selected_tool_ids


def test_weakness_only_query_selects_tool_a_only():
    result = deterministic_tool_selection("Diagnose the Golden State Warriors weaknesses.")

    assert result.selected_tool_ids == (TOOL_A, FINAL_SUMMARY)


def test_unknown_llm_tool_names_are_removed():
    result = validate_tool_selection(["made_up_tool", TOOL_A], {TOOL_A: "Need diagnosis."})

    assert result.selected_tool_ids == (TOOL_A, FINAL_SUMMARY)
    assert any("made_up_tool" in warning for warning in result.warnings)


def test_tool_c_cannot_be_selected_without_tool_b_dependency():
    result = validate_tool_selection([TOOL_C], {TOOL_C: "Rank players."})

    assert TOOL_C in result.selected_tool_ids
    assert TOOL_B in result.selected_tool_ids
    assert TOOL_B in result.required_dependency_ids


def test_sensitivity_cannot_be_selected_without_tool_c_dependency():
    result = validate_tool_selection([SENSITIVITY], {SENSITIVITY: "Check stability."})

    assert SENSITIVITY in result.selected_tool_ids
    assert TOOL_C in result.selected_tool_ids
    assert TOOL_C in result.required_dependency_ids


def test_llm_player_tools_are_removed_for_need_only_query(monkeypatch):
    def fake_llm_call(messages, fallback):
        return LLMCallResult(
            ok=True,
            content={
                "selected_tools": [TOOL_A, NEED_REASONING, TOOL_B, TOOL_C],
                "rationales": {},
                "skipped_tools": {},
            },
            model="test-model",
        )

    monkeypatch.setattr(selector_module, "call_llm_json_result", fake_llm_call)

    result = select_tools_for_query(
        "For the Golden State Warriors, explain which team needs matter most "
        "if the goal is improving interior defense.",
        use_llm=True,
    )

    assert result.selected_tool_ids == (TOOL_A, NEED_REASONING, FINAL_SUMMARY)
    assert TOOL_B not in result.selected_tool_ids
    assert TOOL_C not in result.selected_tool_ids
