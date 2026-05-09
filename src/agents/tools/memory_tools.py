"""Tool functions: read and write player coaching history.

Thin wrappers around PlayerMemoryService that expose a stable ADK-tool
interface (plain Python functions with type-annotated signatures).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level singleton — lazy init to avoid import overhead in CI
_memory_service: Any = None


def _get_memory_service() -> Any:
    global _memory_service
    if _memory_service is None:
        from src.agents.memory import PlayerMemoryService

        _memory_service = PlayerMemoryService()
    return _memory_service


def load_player_history(player_id: str) -> dict[str, Any]:
    """
    Load a player's coaching history for session personalisation.

    Returns session count, recurring issues (seen 2+ times), and
    the five most recently assigned drill names to inform the coaching
    agents before analysis starts.

    Args:
        player_id: Unique player identifier.

    Returns:
        Dict with keys: player_id, total_sessions, recurring_issues,
        recent_drills. Returns zero-state on any error.
    """
    try:
        svc = _get_memory_service()
        session = svc.load(player_id)
        return {
            "player_id": session.player_id,
            "total_sessions": session.total_sessions,
            "recurring_issues": session.recurring_issues,
            "recent_drills": session.recent_drills,
        }
    except Exception as e:
        logger.error("Could not load history for %s: %s", player_id, e)
        return {
            "player_id": player_id,
            "total_sessions": 0,
            "recurring_issues": [],
            "recent_drills": [],
        }


def save_coaching_result(
    player_id: str,
    feedback_json: str,
    player_level: str = "intermediate",
) -> str:
    """
    Persist a coaching feedback result to the player's history.

    Updates issue frequency tracking, drill history, and shot records
    so the next session can personalise coaching based on prior work.

    Args:
        player_id: Unique player identifier.
        feedback_json: JSON string of a CoachingFeedback model.
        player_level: Player skill level string (e.g. "intermediate").

    Returns:
        Confirmation string with the updated session count, or an error message.
    """
    try:
        from src.api.schemas.domain import CoachingFeedback, PlayerLevel

        svc = _get_memory_service()
        level = PlayerLevel(player_level)
        feedback = CoachingFeedback.model_validate(json.loads(feedback_json))
        updated = svc.record_feedback(player_id, feedback, level)
        return f"Saved session #{updated.total_sessions} for player {player_id}"
    except Exception as e:
        logger.error("Could not save result for %s: %s", player_id, e)
        return f"Error saving result: {e}"
