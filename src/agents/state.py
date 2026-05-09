"""Shared state models for the AI Shoot agent pipeline."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Annotated

from pydantic import BaseModel, Field

from src.api.schemas.domain import PlayerLevel, ShotResult


class ShotRecord(BaseModel):
    """Single shot result stored in player history."""

    session_id: str
    timestamp: datetime
    shot_result: ShotResult
    primary_correction: str
    drills_assigned: list[str] = Field(default_factory=list)
    alignment_score: Annotated[float, Field(ge=0.0, le=1.0)] | None = None


class PlayerSession(BaseModel):
    """
    Persistent player state — maps to ADK Memory Bank in production.

    Tracks per-player coaching history to enable adaptive, personalized
    feedback across multiple sessions.
    """

    player_id: str
    player_level: PlayerLevel = PlayerLevel.INTERMEDIATE
    total_sessions: int = 0
    shot_history: list[ShotRecord] = Field(default_factory=list)
    issue_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Frequency map of detected issues across all sessions",
    )
    drills_history: list[str] = Field(
        default_factory=list,
        description="All drill names ever assigned to this player (in order)",
    )

    @property
    def recurring_issues(self) -> list[str]:
        """Issues seen 2+ times, sorted by descending frequency."""
        return [
            issue
            for issue, count in sorted(self.issue_counts.items(), key=lambda x: -x[1])
            if count >= 2
        ]

    @property
    def recent_drills(self) -> list[str]:
        """Last 5 unique drill names assigned (most recent first)."""
        seen: set[str] = set()
        result: list[str] = []
        for drill in reversed(self.drills_history):
            if drill not in seen:
                seen.add(drill)
                result.append(drill)
            if len(result) >= 5:
                break
        return result
