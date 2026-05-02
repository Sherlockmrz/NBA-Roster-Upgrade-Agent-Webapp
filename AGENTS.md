# AGENTS.md

Instructions for future coding agents working in this repository.

## Project scope

This repository is a Python + Streamlit web app for the NBA roster-upgrade analytics pipeline. Keep the repo focused on making the original NBA roster-upgrade agent understandable, runnable, and polished as a web app.

Preferred run command:

```bash
python -m streamlit run app/app.py
```

Preferred test command:

```bash
pytest
```

## Non-negotiable project rules

- Preserve the original Tool A / Tool B / Tool C analytics logic and ordering.
- Use the same original dataset style from the previous NBA-Roster-Upgrade-Agent project.
- Do not introduce a new salary dataset.
- Do not pretend salary, contracts, injuries, or trade rumors are available unless they are explicitly present in the current dataset and code path.
- If salary is mentioned in the UI, show it as unavailable in the current dataset.
- Keep deterministic analytics and LLM reasoning separated.
- The app must work without an API key through fallback deterministic mode.
- Never hardcode API keys, tokens, model credentials, or secrets.
- Use environment variables for OpenRouter configuration.
- Keep `.env` and `.streamlit/secrets.toml` ignored by git.

## Required pipeline order

The Streamlit app must present every stage in this order so a viewer can understand the whole workflow at a glance:

1. User Query
2. Parsed Query
3. Agent Plan
4. Tool A: Team Need Diagnosis
5. LLM Need Reasoning
6. Tool B: Player Strength Representation
7. Tool C: Fit Ranking
8. LLM Feasibility Critique
9. Sensitivity / Robustness Check
10. Final Scouting Summary
11. Grounded Q&A

Do not reorder these stages in the UX, orchestration, README examples, screenshots, or tests unless the user explicitly asks for a redesign.

## Analytics boundaries

- Tool A is responsible for team need diagnosis.
- Tool B is responsible for player strength representation.
- Tool C is responsible for fit ranking.
- Sensitivity / robustness checks should be deterministic and reproducible.
- Numeric metrics, rankings, weights, scores, and player/team facts must come from deterministic code or loaded data.
- Do not use an LLM to fabricate, estimate, or "fill in" missing numerical statistics.
- If a required field is unavailable, surface that limitation clearly instead of inventing a substitute.
- Salary, contract, injury, and trade-rumor data are unavailable unless already present in the accepted dataset. Treat them as unavailable by default.

## LLM boundaries

The LLM may:

- Parse user intent into a structured query.
- Produce an agent plan.
- Explain needs and fit using available analytics outputs.
- Critique feasibility using only available, grounded evidence.
- Summarize deterministic results.
- Answer grounded questions about the shown pipeline outputs.

The LLM must not:

- Invent numerical player, team, salary, contract, injury, or rumor data.
- Change deterministic rankings without a deterministic tool result.
- Hide missing data limitations.
- Require an API key for the app to run.

When no OpenRouter API key is configured, the app should continue in deterministic fallback mode with clear but non-alarming messaging.

## Environment and secrets

- Read OpenRouter settings from environment variables, not source code.
- Recommended variables:
  - `OPENROUTER_API_KEY`
  - `OPENROUTER_MODEL`
  - `OPENROUTER_BASE_URL`
- Keep `.env.example` safe and placeholder-only.
- Do not commit `.env`, `.streamlit/secrets.toml`, notebooks with secrets, logs with tokens, or cached API responses containing private data.

## UI expectations

- Build the real workflow surface, not a marketing page.
- Keep the Tool A / Tool B / Tool C sequence visible and easy to scan.
- Make unavailable data explicit, especially salary, contracts, injuries, and trade rumors.
- Avoid UI copy that implies unsupported data exists.
- Prefer clear Streamlit sections, status blocks, tables, charts, and grounded summaries.
- The app should remain usable in deterministic fallback mode.

## Code organization expectations

- Keep Streamlit UI code under `app/`.
- Keep reusable analytics and orchestration code under `nba_agent/`.
- Keep deterministic tools under `nba_agent/tools/`.
- Keep LLM integration, prompts, parsing, planning, reasoning, and critique under `nba_agent/llm/`.
- Keep data loading and preprocessing under `nba_agent/data/`.
- Keep chart and radar helpers under `nba_agent/visuals/`.
- Keep tests under `tests/`.

Favor small, testable functions for analytics. UI components should call the orchestration layer rather than duplicating pipeline logic.

## Testing expectations

- Run `pytest` after meaningful code changes when dependencies are available.
- Add focused tests when changing deterministic analytics, parsing, ranking, fallback behavior, or unavailable-data handling.
- Tests should verify Tool A / Tool B / Tool C order where orchestration or UI stage metadata is changed.
- If tests cannot run because dependencies are missing, say so clearly in the final response.

## Editing checklist for future agents

Before editing:

- Confirm the change preserves the Tool A / Tool B / Tool C pipeline.
- Check whether the change touches deterministic analytics, LLM behavior, UI presentation, or data contracts.
- Inspect existing code patterns before adding new abstractions.
- Verify that no new salary, contract, injury, or trade-rumor assumptions are being introduced.

While editing:

- Keep deterministic calculations outside LLM modules.
- Keep LLM text grounded in tool outputs and loaded data.
- Preserve fallback deterministic mode when no API key is set.
- Keep secrets out of source files, docs, examples, and tests.
- Use clear "unavailable in current dataset" messaging for unsupported data.

Before finishing:

- Run `pytest` if possible.
- Run or mentally verify `python -m streamlit run app/app.py` remains the intended entrypoint.
- Check that the UI still presents all required stages in order.
- Check `.gitignore` still excludes `.env` and `.streamlit/secrets.toml`.
- Summarize changed files, verification performed, and any remaining placeholders or limitations.
