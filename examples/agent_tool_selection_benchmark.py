"""Run the lightweight agent tool-selection benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nba_agent.evaluation.agent_benchmark import (
    aggregate_metrics_to_dataframe,
    run_agent_tool_selection_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether the agentic planner selects the expected tools."
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Attempt OpenRouter tool selection. Falls back safely if unavailable.",
    )
    args = parser.parse_args()

    result = run_agent_tool_selection_benchmark(use_llm=args.use_llm)
    aggregate_df = aggregate_metrics_to_dataframe(result.aggregate_metrics)
    detail_df = result.to_dataframe()

    print("Agent Tool-Selection Benchmark")
    print("=" * 38)
    print("\nAggregate metrics")
    print(aggregate_df.to_string(index=False))
    print("\nCase-level results")
    print(detail_df.to_string(index=False))


if __name__ == "__main__":
    main()
