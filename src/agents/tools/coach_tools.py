"""Tool function: generate personalized basketball coaching feedback via VLM.

Wraps the AI Shoot Phase 2 BasketballVLMAnalyzer (Gemini Flash).
Uses deferred imports so the module loads in CI without API keys.
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
        JSON-serializable dict matching CoachingFeedback schema.
        Falls back to a heuristic response if VLM is unavailable.
    """
    try:
        bio_data = json.loads(biomechanics_json)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid biomechanics JSON: {e}"}

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
        return feedback.model_dump()
    except ImportError:
        logger.warning("VLM modules not available — returning stub coaching feedback")
        return _stub_coaching_feedback(bio_data, player_id)
    except Exception as e:
        logger.error("VLM coaching failed: %s", e)
        return _stub_coaching_feedback(bio_data, player_id, error=str(e))


def _stub_coaching_feedback(
    bio_data: dict[str, Any],
    player_id: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Heuristic fallback when VLM is unavailable."""
    primary = bio_data.get("primary_issue") or "Focus on your release mechanics"
    return {
        "player_id": player_id,
        "shot_result": bio_data.get("shot_result", "unknown"),
        "confidence": 0.0,
        "summary": f"Analysis complete. {primary}",
        "primary_correction": primary,
        "detailed_analysis": error or "VLM unavailable — biomechanics-only analysis.",
        "biomechanics": bio_data,
        "drills": [],
        "model_used": "fallback",
        "processing_time_ms": None,
    }
