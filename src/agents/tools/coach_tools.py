"""Tool function: generate personalized basketball coaching feedback via VLM.

Wraps the AI Shoot Phase 2 BasketballVLMAnalyzer (Gemini Flash).
Uses deferred imports so the module loads in CI without VLM SDK.

Failure mode (T0-5 course-correction): when the VLM call fails for any
reason (missing API key, network outage, safety-filter block, …) this
function returns ``{"error": "<reason>"}`` **without** a ``summary`` key,
so the orchestrator can detect the failure and surface ``status=error``
to the API caller. Returning a heuristic stub with a populated ``summary``
caused the orchestrator to report ``status=done`` for a degraded analysis
(Bug B) and leaked the raw exception text into the user-facing
``detailed_analysis`` field (Bug A).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_coaching_feedback(
    biomechanics_json: str,
    player_id: str = "anonymous",
    player_level: str = "intermediate",
    session_count: int = 0,
    recurring_issues_json: str = "[]",
    previous_drills_json: str = "[]",
) -> dict[str, Any]:
    """
    Generate personalized basketball coaching feedback using Gemini VLM.

    Combines structured biomechanics data with player history to produce
    adaptive, level-appropriate coaching in natural language.

    Args:
        biomechanics_json: JSON string matching BiomechanicsReport schema.
        player_id: Unique player identifier for personalization.
        player_level: Skill level — beginner/intermediate/advanced/elite.
        session_count: Number of previous sessions (personalizes tone).
        recurring_issues_json: JSON list of recurring issues to address first.
        previous_drills_json: JSON list of drills to avoid repeating.

    Returns:
        On success: dict matching CoachingFeedback schema (model_dump).
        On failure: ``{"error": "<reason>", "player_id": "..."}`` — no
        ``summary`` key, so the orchestrator propagates the failure.
    """
    try:
        bio_data = json.loads(biomechanics_json)
    except json.JSONDecodeError:
        return {"error": "invalid_biomechanics_json", "player_id": player_id}

    try:
        recurring_issues: list[str] = json.loads(recurring_issues_json)
        previous_drills: list[str] = json.loads(previous_drills_json)
    except json.JSONDecodeError:
        recurring_issues = []
        previous_drills = []

    try:
        from src.api.schemas.domain import BiomechanicsReport, PlayerLevel, SessionContext
        from src.vlm.basketball_analyzer import BasketballVLMAnalyzer
        from src.vlm.gemini_client import GeminiFlashClient
    except ImportError as e:
        # VLM SDK not installed — config / env error, not user-input. Stable
        # code in the response; the SDK name goes to logs only.
        logger.error("VLM modules unavailable: %s", e)
        return {"error": "vlm_unavailable", "player_id": player_id}

    try:
        report = BiomechanicsReport.model_validate(bio_data)
        context = SessionContext(
            player_id=player_id,
            player_level=PlayerLevel(player_level),
            session_count=session_count,
            recurring_issues=recurring_issues,
            previous_drills=previous_drills,
        )
        client = GeminiFlashClient.from_env()
        analyzer = BasketballVLMAnalyzer(client)
        feedback = analyzer.analyze(report, context)
        return feedback.model_dump(mode="json")
    except Exception as e:
        # Logged with full traceback for ops; response stays generic so we
        # never leak internal state (API keys, paths, stack frames) to the
        # user via the response body.
        logger.exception("VLM coaching failed for player %s", player_id)
        return {"error": f"coaching_failed:{type(e).__name__}", "player_id": player_id}
