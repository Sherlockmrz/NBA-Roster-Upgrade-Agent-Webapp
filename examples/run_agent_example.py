"""Runnable deterministic roster-agent example.

Run from the repository root:

    python examples/run_agent_example.py
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nba_agent.agent import run_roster_agent


def main() -> None:
    user_query = (
        "Recommend top 5 players for the Warriors to improve interior defense "
        "using the last 10 games."
    )

    filters = {
        "team": "Warriors",
        "goal": "interior defense",
        "top_k": 5,
        "recent_games": 10,
        "min_games": 15,
        "min_avg_minutes": 15,
        "exclude_current_team": True,
        "ranking_mode": "Best Talent",
    }

    result = run_roster_agent(
        user_query=user_query,
        filters=filters,
        data_dir="data/raw",
        use_llm=False,
    )

    print("Warnings:")
    if result.warnings:
        for warning in result.warnings:
            print(f"- {warning}")
    else:
        print("- None")

    print("\nTrace:")
    for step in result.trace_steps:
        print(f"{step.step_number}. {step.title} [{step.status}]")

    print("\nTop recommendations:")
    print(result.ranked_df.to_string(index=False))

    print("\nFinal summary:")
    print(result.final_summary)


if __name__ == "__main__":
    main()
