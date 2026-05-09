"""Player history routes — GET /player/{player_id}/history."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from src.agents.memory import PlayerMemoryService
from src.api.schemas.responses import PlayerHistoryResponse, SessionHistoryItem

router = APIRouter(prefix="/player", tags=["players"])
logger = logging.getLogger(__name__)


@router.get(
    "/{player_id}/history",
    response_model=PlayerHistoryResponse,
    summary="Retrieve player coaching history",
)
async def get_player_history(player_id: str) -> PlayerHistoryResponse:
    """
    Return the full coaching history for a player.

    Includes session count, recurring issues (seen ≥2×), and the 20 most
    recent shot records sorted newest-first. Returns an empty history (not 404)
    for players with no prior sessions.
    """
    svc = PlayerMemoryService()
    session = svc.load(player_id)

    recent_sessions = [
        SessionHistoryItem(
            session_id=record.session_id,
            shot_result=record.shot_result.value,
            primary_correction=record.primary_correction,
            timestamp=record.timestamp,
            alignment_score=record.alignment_score,
            drills_assigned=record.drills_assigned,
        )
        for record in reversed(session.shot_history[-20:])
    ]

    return PlayerHistoryResponse(
        player_id=player_id,
        player_level=session.player_level,
        session_count=session.total_sessions,
        recurring_issues=session.recurring_issues,
        recent_sessions=recent_sessions,
    )
