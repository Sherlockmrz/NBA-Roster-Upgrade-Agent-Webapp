"""Zero-shot LLM baseline for evaluation only."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from nba_agent.llm.client import call_llm_json_result, get_llm_status


@dataclass(frozen=True)
class ZeroShotResult:
    """Validated zero-shot recommendation output."""

    players: list[dict[str, Any]]
    limitations: str
    used_fallback: bool
    model: str
    error_type: str = ""
    error_message: str = ""


ZERO_SHOT_SYSTEM_PROMPT = """
You are a zero-shot NBA roster recommendation baseline.

You receive only the natural-language user query. You do not receive Tool A,
Tool B, Tool C, player strength tables, fit scores, salary data, contracts,
injuries, trade rumors, or current NBA news.

Return strict JSON only with this shape:
{
  "players": [
    {"rank": 1, "player_name": "...", "reasoning": "..."}
  ],
  "limitations": "..."
}

This is zero-shot general reasoning. Do not claim access to the project dataset.
Do not invent exact statistics. Do not mention salary unless saying salary data
is not provided.
""".strip()


def run_zero_shot_baseline(
    user_query: str,
    top_k: int,
    use_llm: bool = True,
) -> ZeroShotResult:
    """Return zero-shot LLM recommendations or a clean unavailable fallback."""

    status = get_llm_status()
    if not use_llm or not status.available:
        error_type = "llm_disabled" if not use_llm else "missing_api_key"
        error_message = (
            "Zero-shot baseline requires LLM access."
            if not status.available
            else "Zero-shot baseline was not requested."
        )
        return ZeroShotResult(
            players=[],
            limitations="Zero-shot baseline requires LLM access.",
            used_fallback=True,
            model=status.model,
            error_type=error_type,
            error_message=error_message,
        )

    fallback = {
        "players": [],
        "limitations": "Zero-shot baseline requires LLM access.",
    }
    result = call_llm_json_result(
        [
            {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_query": user_query,
                        "top_k": top_k,
                        "instruction": f"Recommend exactly {top_k} NBA players if possible.",
                    },
                    ensure_ascii=True,
                ),
            },
        ],
        fallback=fallback,
    )
    if not result.ok or not isinstance(result.content, dict):
        return ZeroShotResult(
            players=[],
            limitations=fallback["limitations"],
            used_fallback=True,
            model=result.model,
            error_type=result.error_type,
            error_message=result.error_message,
        )

    return _validate_zero_shot_payload(result.content, top_k, result.model)


def _validate_zero_shot_payload(
    payload: dict[str, Any],
    top_k: int,
    model: str,
) -> ZeroShotResult:
    raw_players = payload.get("players")
    if not isinstance(raw_players, list):
        return ZeroShotResult(
            players=[],
            limitations="Zero-shot response did not include a valid players list.",
            used_fallback=True,
            model=model,
            error_type="json_validation_error",
            error_message="Zero-shot response did not include a valid players list.",
        )

    players: list[dict[str, Any]] = []
    for index, item in enumerate(raw_players[:top_k], start=1):
        if not isinstance(item, dict):
            continue
        player_name = str(item.get("player_name", "")).strip()
        if not player_name:
            continue
        players.append(
            {
                "rank": int(item.get("rank") or index),
                "player_name": player_name,
                "reasoning": str(item.get("reasoning", "")).strip(),
            }
        )

    limitations = str(payload.get("limitations", "")).strip()
    if not limitations:
        limitations = "Zero-shot baseline was generated without project dataset access."

    return ZeroShotResult(
        players=players,
        limitations=limitations,
        used_fallback=False,
        model=model,
    )

