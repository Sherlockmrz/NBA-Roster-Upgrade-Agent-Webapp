# NBA-Roster-Upgrade-Agent-Webapp

A polished Streamlit web app scaffold for the original **NBA-Roster-Upgrade-Agent** workflow.

## Scope (current)

- Preserves the original Tool A / Tool B / Tool C pipeline concept.
- Keeps the original dataset style intent.
- **Does not introduce any new salary dataset.** Salary-dependent feasibility is marked unavailable.

## Run locally

```bash
python -m streamlit run app/app.py
```

## Run the deterministic agent example

```bash
python examples/run_agent_example.py
```

Example usage:

```python
from nba_agent.agent import run_roster_agent

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

print(result.ranked_df)
print(result.final_summary)
```

If optional filters are missing, `run_roster_agent` uses deterministic fallback defaults and records warnings in `result.warnings`. Salary data is unavailable in the current dataset.

## Status

The deterministic Tool A / Tool B / Tool C analytics modules and main agent orchestration are implemented. LLM orchestration remains placeholder-only and the app can still run without an API key.
