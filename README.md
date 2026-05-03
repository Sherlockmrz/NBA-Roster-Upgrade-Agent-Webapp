# NBA Roster Upgrade Agent WebApp

An explainable LLM-powered front-office assistant for NBA roster diagnosis, player fit ranking, robustness checking, and grounded scouting Q&A.

**Streamlit WebApp** · **LLM Agent Reasoning** · **Explainable Sports Analytics** · **Deterministic Fallback** · **Grounded Q&A**



![WebApp Home](assets/webapp_home.png)



## Overview

NBA Roster Upgrade Agent WebApp turns a natural-language roster question into an auditable player recommendation workflow. A user can ask for roster upgrades for a specific team and goal, such as improving interior defense, and the app walks through every stage: parsing the request, diagnosing team needs, reasoning about which needs should matter most, ranking player fits, checking recommendation robustness, summarizing the result, and answering grounded follow-up questions. The product goal is not to hide a ranking model behind a single score. It is to make the recommendation trace visible enough for a viewer to understand why each player appears.

## Why This Project Matters

NBA roster decisions are multi-factor. Teams care about current weaknesses, player strengths, role fit, sample size, robustness, and data availability. Pure ranking models can be difficult to trust because they compress the entire recommendation into one number. This app combines deterministic analytics with bounded LLM reasoning so the user can inspect the chain of evidence instead of only seeing an answer.

The key design choice is separation of responsibilities:

| Layer | Responsibility |
| --- | --- |
| Deterministic tools | Calculate team needs, player strengths, fit scores, and robustness checks. |
| LLM layers | Parse intent, explain tactical emphasis, summarize computed outputs, and answer grounded questions. |
| UI | Show the whole workflow as an explainable product demo with cards, charts, tables, and chat. |

The LLM is not allowed to invent stats, salary, contracts, injuries, trade rumors, or current NBA news. If those fields are not in the current dataset, the app marks them unavailable.

## Example User Query

```text
Recommend top 5 players for the Golden State Warriors to improve interior defense using the last 10 games. Only include players with at least 15 games and 15 average minutes.
```

## What The App Returns

After the agent runs, the app displays:

- Parsed query fields such as team, goal, top K, recent games, minimum games, and minimum average minutes.
- Team need diagnosis from Tool A.
- Bounded LLM Need Reasoning that adjusts need weights without changing raw stats.
- Top player recommendation cards with fit score, best match, profile text, image placeholder, salary unavailable placeholder, and ability radar chart.
- Player strength vectors from Tool B.
- Fit ranking from Tool C.
- Sensitivity / Robustness Check showing whether top recommendations remain stable after small need-weight perturbations.
- Final Scouting Summary generated from computed outputs or deterministic fallback.
- Grounded Q&A that answers only from the current agent result.

## Agent Pipeline

![Agent Pipeline](assets/agent_trace.png)

## Detailed Workflow

### Step 1: User Query

**What it does:** Captures the natural-language roster question from the user.

**Input:** A free-form roster request.

**Output:** The exact user query passed into the agent.

**Why it is useful:** The query is the first source of truth. The app is designed around query-first parsing, so the sidebar follows the parsed query unless manual overrides are enabled.

### Step 2: Parsed Query

**What it does:** Converts the natural-language request into structured fields.

**Input:** User query plus default sidebar values.

**Output:** Team, goal, top K, recent games, min games, min average minutes, exclude-current-team setting, ranking mode, and unavailable constraints.

**Why it is useful:** It makes the user’s request inspectable before analytics run. If LLM parsing is disabled or unavailable, deterministic parsing still works.

### Step 3: Agent Plan

**What it does:** Displays the ordered workflow used by the app.

**Input:** Parsed query and available tools.

**Output:** A display-friendly plan that preserves the Tool A / Tool B / Tool C order.

**Why it is useful:** The user can see the execution sequence before reading individual outputs. The current app keeps this plan deterministic and does not allow an LLM to reorder the required analytics tools.

### Step 4: Tool A - Team Need Diagnosis

**What it does:** Diagnoses recent team weaknesses and converts them into need weights.

**Input:** Team, recent games window, raw team/game data.

**Output:** `need_df` with metrics, labels, need weights, and goal-boosted flags where applicable.

**Why it is useful:** It grounds the recommendation in what the selected team appears to lack in the loaded dataset.

### Step 5: LLM Need Reasoning

**What it does:** Interprets the basketball goal and recommends bounded multipliers for existing Tool A metrics.

**Input:** Parsed goal, Tool A need table, and allowed metric names.

**Output:** Tactical interpretation, metric multipliers, explanations, and `adjusted_need_df`.

**Why it is useful:** It adds an explainable tactical layer without letting the LLM calculate player rankings or invent metrics.

### Step 6: Tool B - Player Strength Representation

**What it does:** Builds candidate player strength vectors from the box-score dataset.

**Input:** Player game data plus filters such as min games, min average minutes, and exclude-current-team.

**Output:** `player_strength_df` with player-level strength features and radar-display fields.

**Why it is useful:** It creates comparable player profiles that Tool C can score against team needs.

### Step 7: Tool C - Fit Ranking

**What it does:** Scores and ranks players by fit.

**Input:** Adjusted need weights and player strength vectors.

**Output:** Ranked players with fit score and best-match explanation.

**Why it is useful:** It produces the core recommendation list while keeping the score traceable to deterministic inputs.

### Step 8: Sensitivity / Robustness Check

**What it does:** Perturbs adjusted need weights by a small amount and checks whether top recommendations remain similar.

**Input:** Adjusted need weights, player strength vectors, and original ranking.

**Output:** Stability label, top-k overlap, original top players, perturbed top players, explanation, and rank comparison table.

**Why it is useful:** It helps distinguish stable recommendations from rankings that depend too heavily on small weight changes.

### Step 9: Final Scouting Summary

**What it does:** Summarizes the computed pipeline outputs in concise scouting language.

**Input:** Parsed query, agent plan, need tables, need reasoning, ranked players, and sensitivity output.

**Output:** Executive summary, three key takeaways, limitations note, fallback status, and warnings.

**Why it is useful:** It gives a dashboard-friendly readout without requiring the viewer to inspect every table.

### Step 10: Grounded Q&A

**What it does:** Lets the user ask follow-up questions about the current run.

**Input:** User chat question and current `AgentResult`.

**Output:** A grounded answer or deterministic fallback answer.

**Why it is useful:** It turns the app into an interactive explanation surface. The assistant can answer questions about the current pipeline outputs, but it cannot search the web or invent unavailable facts.

## How LLM Is Used

The LLM is used as a bounded reasoning layer, not as the source of numerical truth.

### LLM Is Used For

- Parsing natural-language roster requests when `Use LLM` is enabled and an API key is available.
- Tactical need reasoning over existing Tool A metrics.
- Final scouting summary generation from computed outputs.
- Grounded Q&A over the current `AgentResult`.

The current agent plan display is deterministic and preserves the required Tool A / Tool B / Tool C order. LLMs are not allowed to reorder the core analytics pipeline.

### LLM Is Not Used For

- Inventing player stats.
- Replacing Tool A / Tool B / Tool C calculations.
- Estimating salary.
- Reporting injuries.
- Describing contracts.
- Surfacing trade rumors.
- Adding current NBA news.
- Filling in unavailable advanced stats.

If LLM calls fail, return invalid JSON, hit rate limits, or no API key is configured, the app falls back to deterministic behavior.

## Tool Methodology

### Tool A: Team Need Diagnosis

Tool A looks at recent team performance and identifies relative weaknesses using a z-score style approach. The output is a need table where higher need weights mean the selected team has a stronger computed need in that category.

At a high level:

```text
Team metric weakness -> normalized need signal -> need_weight
```

The app displays top weakness cards and a need-weight chart so the user can see what the ranking is trying to solve.

### LLM Need Reasoning

LLM Need Reasoning receives only:

- Parsed goal.
- Tool A need rows.
- Allowed metric names and labels.

It may recommend multipliers only for metrics already present in the need table. Multipliers are bounded and validated.

```text
Adjusted Need Weight = Original Need Weight × LLM Multiplier
```

If the LLM is unavailable or returns invalid output, deterministic fallback rules are used. For example, an interior-defense goal can emphasize rebounding and rim protection when those metrics exist.

### Tool B: Player Strength Representation

Tool B aggregates player performance into strength vectors. These vectors are based on available dataset columns and filters such as minimum games and minimum average minutes.

Example strength dimensions include:

- Rebounding.
- Rim protection.
- Perimeter defense.
- Playmaking.
- Three-point shooting.
- Scoring, when available for radar display.

### Tool C: Fit Ranking

Tool C ranks players by matching team needs against player strengths.

```text
Fit Score = Need Weights · Player Strength Vector
```

When adjusted need weights are available, Tool C uses them for ranking. The LLM does not directly calculate fit scores.

### Sensitivity Analysis

The Sensitivity / Robustness Check perturbs adjusted need weights by a small amount and recomputes or simulates the ranking. It then compares the original and perturbed top recommendations.

Interpretation:

| Label | Meaning |
| --- | --- |
| Stable | The top recommendation set remains highly similar after perturbation. |
| Somewhat Stable | Some top players remain, but the ranking changes meaningfully. |
| Unstable | Small weight changes substantially alter the top recommendations. |

## UI Features

- Natural-language query input.
- Prominent `Run Agent` button directly under the query input.
- `Use LLM` toggle beside the run button.
- Compact model status display showing LLM mode, model name, API key availability, and runtime mode.
- Query-first parsing with sidebar synchronization.
- Optional manual sidebar overrides.
- Top 5 recommendation cards before the detailed trace.
- Player image placeholder.
- Salary unavailable placeholder.
- Ability radar / hexagon-style chart for recommended players.
- Detailed workflow trace with every stage visible.
- Charts for need weights, adjusted need weights, fit scores, and robustness outputs.
- Expanders for full data tables to avoid dumping large raw dataframes.
- Grounded Q&A after the agent has run.

## Data

This app uses the existing dataset style from the original NBA roster-upgrade project. Required raw files should be placed under `data/raw/`:

```text
data/raw/teams.csv
data/raw/games.csv
data/raw/games_details.csv
```

Optional raw files may also exist depending on the local dataset copy:

```text
data/raw/players.csv
data/raw/ranking.csv
```

The current app does not include salary, contract, injury, trade-rumor, real-time roster, or current NBA news data. Any request for those fields is surfaced as unavailable instead of being invented.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Sherlockmrz/NBA-Roster-Upgrade-Agent-Webapp.git
cd NBA-Roster-Upgrade-Agent-Webapp
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Data Setup

Place the original NBA dataset CSV files in `data/raw/`:

```text
data/raw/teams.csv
data/raw/games.csv
data/raw/games_details.csv
```

There is no Kaggle download script currently included in this repository. Use the same dataset style as the original NBA Roster Upgrade Agent project and keep the filenames above.

## Environment Variables

Copy `.env.example` to `.env` for local development:

```bash
cp .env.example .env
```

Then fill in local values as needed:

```env
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openrouter/free
OPENROUTER_SITE_URL=https://github.com/Sherlockmrz/NBA-Roster-Upgrade-Agent-Webapp
OPENROUTER_APP_NAME=NBA Roster Upgrade Agent WebApp
```

Notes:

- `.env` must not be committed.
- `.env.example` contains placeholders only.
- The app works without an API key using deterministic fallback mode.
- The default model is `openrouter/free` when `OPENROUTER_MODEL` is missing.
- The UI never displays the full API key.

## Run Locally

```bash
python -m streamlit run app/app.py
```

Open the local Streamlit URL shown in the terminal.

## Run Tests

```bash
pytest
```

Run the deterministic example:

```bash
python examples/run_agent_example.py
```

## Programmatic Usage

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

## Project Structure

```text
NBA-Roster-Upgrade-Agent-Webapp/
├── app/
│   ├── app.py
│   ├── components.py
│   └── style.css
├── nba_agent/
│   ├── data/
│   ├── tools/
│   ├── llm/
│   ├── visuals/
│   ├── agent.py
│   └── schemas.py
├── data/
│   └── raw/
├── examples/
│   └── run_agent_example.py
├── notebooks/
├── tests/
├── assets/
├── README.md
├── requirements.txt
└── .env.example
```

## Current Limitations

- Salary data is unavailable in the current dataset.
- Contract data is unavailable.
- Injury data is unavailable.
- Trade rumors and current NBA news are unavailable.
- There is no real trade simulator.
- Feasibility critique is not currently implemented.
- Recommendations depend on the available dataset columns.
- LLM outputs are validated and fallback-protected, but they still require careful interpretation.
- Player images are placeholders for now.
- The app is not a live roster transaction system.

## Future Improvements

- Add a properly sourced salary and contract dataset.
- Add a real feasibility or trade critique module.
- Add player images and richer player profile metadata.
- Add a deployed public app link.
- Add richer player archetype and role descriptions.
- Add model comparison controls for LLM behavior.
- Add multi-team trade constraints.
- Add stronger benchmark evaluation against historical roster decisions.

## Security Note

OpenRouter settings are read from environment variables through `.env` during local development. Never commit `.env`, API keys, tokens, model credentials, logs containing secrets, or cached private API responses.

Safe files:

- `.env.example` with placeholders.
- README documentation without real secrets.
- Tests that mock or disable API calls.

Unsafe files:

- `.env` with a real `OPENROUTER_API_KEY`.
- Screenshots or logs that reveal private keys.
- Notebooks containing copied tokens.

## Author

**Ruize Ma / Sherlockmrz**

GitHub: [Sherlockmrz](https://github.com/Sherlockmrz)

