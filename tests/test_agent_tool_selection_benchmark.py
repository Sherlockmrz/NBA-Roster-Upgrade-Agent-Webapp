from nba_agent.agentic.tool_registry import (
    FINAL_SUMMARY,
    GROUNDED_QA,
    NEED_REASONING,
    SENSITIVITY,
    TOOL_A,
    TOOL_B,
    TOOL_C,
    ZERO_SHOT_EVALUATION,
)
from nba_agent.evaluation.agent_benchmark import (
    check_dependency_validity,
    compute_selection_metrics,
    default_tool_selection_benchmark_cases,
    run_agent_tool_selection_benchmark,
)


def test_benchmark_cases_include_required_examples():
    cases = default_tool_selection_benchmark_cases()
    by_id = {case.query_id: case for case in cases}

    assert len(cases) >= 8
    assert by_id["case_01_weakness_diagnosis"].expected_tool_ids == (
        TOOL_A,
        FINAL_SUMMARY,
    )
    assert by_id["case_02_need_reasoning"].expected_tool_ids == (
        TOOL_A,
        NEED_REASONING,
        FINAL_SUMMARY,
    )
    assert by_id["case_03_recommendation"].expected_tool_ids == (
        TOOL_A,
        NEED_REASONING,
        TOOL_B,
        TOOL_C,
        FINAL_SUMMARY,
    )
    assert SENSITIVITY in by_id["case_04_recommendation_robustness"].expected_tool_ids
    assert GROUNDED_QA in by_id["case_05_recommendation_qa"].expected_tool_ids
    assert by_id["case_06_zero_shot_comparison"].expected_tool_ids == (
        ZERO_SHOT_EVALUATION,
        FINAL_SUMMARY,
    )


def test_deterministic_benchmark_runs_and_computes_aggregate_metrics():
    result = run_agent_tool_selection_benchmark(use_llm=False)

    assert result.rows
    assert result.aggregate_metrics["average_exact_match"] == 1.0
    assert result.aggregate_metrics["average_precision"] == 1.0
    assert result.aggregate_metrics["average_recall"] == 1.0
    assert result.aggregate_metrics["average_f1"] == 1.0
    assert result.aggregate_metrics["dependency_validity_rate"] == 1.0
    assert result.aggregate_metrics["execution_safety_rate"] == 1.0
    assert not result.to_dataframe().empty


def test_selection_metrics_handle_empty_selected_tools_safely():
    metrics = compute_selection_metrics([], [TOOL_A, FINAL_SUMMARY])

    assert metrics["exact_match"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_dependency_validity_checks_required_dependencies():
    assert check_dependency_validity("Rank players.", [TOOL_C, FINAL_SUMMARY]) is False
    assert check_dependency_validity("Build strengths.", [TOOL_B, FINAL_SUMMARY]) is False
    assert check_dependency_validity("Check stability.", [SENSITIVITY, FINAL_SUMMARY]) is False
    assert check_dependency_validity("Show summary.", [TOOL_A]) is False


def test_grounded_qa_requires_qa_intent():
    assert check_dependency_validity("Recommend players.", [TOOL_A, FINAL_SUMMARY, GROUNDED_QA]) is False
    assert check_dependency_validity(
        "Recommend players and keep follow-up questions.",
        [TOOL_A, FINAL_SUMMARY, GROUNDED_QA],
    ) is True
