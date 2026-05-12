"""Lightweight benchmark for agentic tool-selection behavior."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

import pandas as pd

from nba_agent.agentic.tool_registry import (
    FINAL_SUMMARY,
    GROUNDED_QA,
    NEED_REASONING,
    SENSITIVITY,
    TOOL_A,
    TOOL_B,
    TOOL_C,
    TOOL_REGISTRY,
    ZERO_SHOT_EVALUATION,
)
from nba_agent.agentic.tool_selector import select_tools_for_query


@dataclass(frozen=True)
class ToolSelectionBenchmarkCase:
    """One expected tool-selection scenario."""

    query_id: str
    query: str
    expected_tool_ids: tuple[str, ...]


@dataclass(frozen=True)
class ToolSelectionBenchmarkRow:
    """Metric output for one tool-selection case."""

    query_id: str
    query: str
    expected_tool_ids: tuple[str, ...]
    selected_tool_ids: tuple[str, ...]
    exact_match: float
    precision: float
    recall: float
    f1: float
    dependency_valid: bool
    execution_safe: bool
    notes: str = ""


@dataclass(frozen=True)
class ToolSelectionBenchmarkResult:
    """Benchmark rows and aggregate metrics."""

    rows: list[ToolSelectionBenchmarkRow]
    aggregate_metrics: dict[str, float]

    def to_dataframe(self) -> pd.DataFrame:
        return benchmark_rows_to_dataframe(self.rows)


def default_tool_selection_benchmark_cases() -> list[ToolSelectionBenchmarkCase]:
    """Return representative prompts with expected display-tool selections."""

    return [
        ToolSelectionBenchmarkCase(
            query_id="case_01_weakness_diagnosis",
            query="Diagnose the main weaknesses of the Golden State Warriors over the last 10 games.",
            expected_tool_ids=(TOOL_A, FINAL_SUMMARY),
        ),
        ToolSelectionBenchmarkCase(
            query_id="case_02_need_reasoning",
            query=(
                "For the Golden State Warriors, explain which team needs matter most "
                "if the goal is improving interior defense."
            ),
            expected_tool_ids=(TOOL_A, NEED_REASONING, FINAL_SUMMARY),
        ),
        ToolSelectionBenchmarkCase(
            query_id="case_03_recommendation",
            query=(
                "Recommend the top 5 players for the Golden State Warriors to improve "
                "interior defense over the last 10 games."
            ),
            expected_tool_ids=(TOOL_A, NEED_REASONING, TOOL_B, TOOL_C, FINAL_SUMMARY),
        ),
        ToolSelectionBenchmarkCase(
            query_id="case_04_recommendation_robustness",
            query=(
                "Recommend the top 5 players for the Golden State Warriors to improve "
                "interior defense and check whether the ranking is robust."
            ),
            expected_tool_ids=(
                TOOL_A,
                NEED_REASONING,
                TOOL_B,
                TOOL_C,
                SENSITIVITY,
                FINAL_SUMMARY,
            ),
        ),
        ToolSelectionBenchmarkCase(
            query_id="case_05_recommendation_qa",
            query="Recommend top 5 players and keep a grounded Q&A section for follow-up questions.",
            expected_tool_ids=(
                TOOL_A,
                NEED_REASONING,
                TOOL_B,
                TOOL_C,
                FINAL_SUMMARY,
                GROUNDED_QA,
            ),
        ),
        ToolSelectionBenchmarkCase(
            query_id="case_06_zero_shot_comparison",
            query="Compare the tool pipeline recommendations against a zero-shot LLM baseline.",
            expected_tool_ids=(ZERO_SHOT_EVALUATION, FINAL_SUMMARY),
        ),
        ToolSelectionBenchmarkCase(
            query_id="case_07_skill_need_reasoning",
            query="Which player skills matter most for improving shooting?",
            expected_tool_ids=(TOOL_A, NEED_REASONING, FINAL_SUMMARY),
        ),
        ToolSelectionBenchmarkCase(
            query_id="case_08_direct_fit_ranking",
            query="Rank candidate players by fit score for the Warriors.",
            expected_tool_ids=(TOOL_A, TOOL_B, TOOL_C, FINAL_SUMMARY),
        ),
        ToolSelectionBenchmarkCase(
            query_id="case_09_perimeter_defense_needs",
            query="Explain which team needs matter most for improving perimeter defense.",
            expected_tool_ids=(TOOL_A, NEED_REASONING, FINAL_SUMMARY),
        ),
        ToolSelectionBenchmarkCase(
            query_id="case_10_followup_without_recommendation",
            query="Diagnose Warriors weaknesses and let me ask follow-up questions.",
            expected_tool_ids=(TOOL_A, FINAL_SUMMARY, GROUNDED_QA),
        ),
    ]


def run_agent_tool_selection_benchmark(
    use_llm: bool = False,
    cases: list[ToolSelectionBenchmarkCase] | None = None,
) -> ToolSelectionBenchmarkResult:
    """Run the tool-selection benchmark and return row-level plus aggregate metrics."""

    benchmark_cases = cases or default_tool_selection_benchmark_cases()
    rows = [evaluate_tool_selection_case(case, use_llm=use_llm) for case in benchmark_cases]
    return ToolSelectionBenchmarkResult(
        rows=rows,
        aggregate_metrics=aggregate_benchmark_metrics(rows),
    )


def evaluate_tool_selection_case(
    case: ToolSelectionBenchmarkCase,
    use_llm: bool = False,
) -> ToolSelectionBenchmarkRow:
    """Evaluate one tool-selection case safely."""

    try:
        selection = select_tools_for_query(case.query, use_llm=use_llm)
        selected = tuple(selection.selected_tool_ids)
        execution_safe = True
        notes = "; ".join(selection.warnings)
    except Exception as exc:  # pragma: no cover - defensive safety path
        selected = ()
        execution_safe = False
        notes = f"selector error: {exc}"

    metrics = compute_selection_metrics(selected, case.expected_tool_ids)
    dependency_valid = check_dependency_validity(case.query, selected)
    return ToolSelectionBenchmarkRow(
        query_id=case.query_id,
        query=case.query,
        expected_tool_ids=case.expected_tool_ids,
        selected_tool_ids=selected,
        exact_match=metrics["exact_match"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        dependency_valid=dependency_valid,
        execution_safe=execution_safe,
        notes=notes,
    )


def compute_selection_metrics(
    selected_tool_ids: tuple[str, ...] | list[str] | set[str],
    expected_tool_ids: tuple[str, ...] | list[str] | set[str],
) -> dict[str, float]:
    """Compute exact match, precision, recall, and F1 for tool sets."""

    selected = set(selected_tool_ids)
    expected = set(expected_tool_ids)
    correct = selected & expected
    precision = len(correct) / len(selected) if selected else 0.0
    recall = len(correct) / len(expected) if expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "exact_match": 1.0 if selected == expected else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def check_dependency_validity(
    query: str,
    selected_tool_ids: tuple[str, ...] | list[str] | set[str],
) -> bool:
    """Return whether selected tools satisfy simple safety dependencies."""

    selected = set(selected_tool_ids)
    if FINAL_SUMMARY not in selected:
        return False
    if TOOL_C in selected and TOOL_B not in selected:
        return False
    if TOOL_B in selected and TOOL_A not in selected:
        return False
    if SENSITIVITY in selected and TOOL_C not in selected:
        return False
    if GROUNDED_QA in selected and not _query_asks_for_qa(query):
        return False
    return True


def aggregate_benchmark_metrics(rows: list[ToolSelectionBenchmarkRow]) -> dict[str, float]:
    """Aggregate row metrics with safe defaults."""

    if not rows:
        return {
            "average_exact_match": 0.0,
            "average_precision": 0.0,
            "average_recall": 0.0,
            "average_f1": 0.0,
            "dependency_validity_rate": 0.0,
            "execution_safety_rate": 0.0,
        }
    return {
        "average_exact_match": mean(row.exact_match for row in rows),
        "average_precision": mean(row.precision for row in rows),
        "average_recall": mean(row.recall for row in rows),
        "average_f1": mean(row.f1 for row in rows),
        "dependency_validity_rate": mean(1.0 if row.dependency_valid else 0.0 for row in rows),
        "execution_safety_rate": mean(1.0 if row.execution_safe else 0.0 for row in rows),
    }


def benchmark_rows_to_dataframe(rows: list[ToolSelectionBenchmarkRow]) -> pd.DataFrame:
    """Return a display-friendly benchmark table."""

    return pd.DataFrame(
        [
            {
                "query_id": row.query_id,
                "query": row.query,
                "expected_tools": ", ".join(tool_names(row.expected_tool_ids)),
                "selected_tools": ", ".join(tool_names(row.selected_tool_ids)),
                "exact_match": row.exact_match,
                "precision": row.precision,
                "recall": row.recall,
                "f1": row.f1,
                "dependency_valid": row.dependency_valid,
                "execution_safe": row.execution_safe,
                "notes": row.notes,
            }
            for row in rows
        ]
    )


def tool_names(tool_ids: tuple[str, ...] | list[str] | set[str]) -> list[str]:
    """Map tool ids to UI names in registry order when possible."""

    ordered_ids = [tool_id for tool_id in TOOL_REGISTRY if tool_id in set(tool_ids)]
    return [TOOL_REGISTRY[tool_id].name for tool_id in ordered_ids]


def aggregate_metrics_to_dataframe(metrics: dict[str, float]) -> pd.DataFrame:
    """Return aggregate metrics as a two-column dataframe."""

    return pd.DataFrame(
        [
            {"metric": metric.replace("_", " ").title(), "value": value}
            for metric, value in metrics.items()
        ]
    )


def _query_asks_for_qa(query: str) -> bool:
    lowered = str(query or "").lower()
    return any(
        term in lowered
        for term in ("q&a", "qa", "follow-up", "follow up", "chat", "ask questions", "further information")
    )
