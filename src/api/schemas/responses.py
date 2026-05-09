"""API request/response schemas (HTTP layer)."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.api.schemas.domain import CoachingFeedback, PlayerLevel  # noqa: TC001


class TaskResponse(BaseModel):
    """Response from POST /analyze — returned immediately (202 Accepted)."""

    task_id: str
    status: Literal["processing", "done", "error"] = "processing"
    player_id: str | None = None
    created_at: datetime


class SessionResponse(BaseModel):
    """Response from GET /session/{task_id} — full task lifecycle view."""

    task_id: str
    status: Literal["processing", "done", "error"]
    player_id: str | None = None
    created_at: datetime
    updated_at: datetime
    result: dict[str, Any] | None = Field(None, description="Pipeline output when status=done")
    error: str | None = Field(None, description="Error message when status=error")
    processing_time_ms: float | None = None


class SessionHistoryItem(BaseModel):
    session_id: str
    shot_result: str
    primary_correction: str
    timestamp: datetime
    alignment_score: float | None = None
    drills_assigned: list[str] = Field(default_factory=list)


class PlayerHistoryResponse(BaseModel):
    player_id: str
    player_level: PlayerLevel
    session_count: int
    recurring_issues: list[str]
    recent_sessions: list[SessionHistoryItem] = Field(default_factory=list)


class HealthComponent(BaseModel):
    status: Literal["ok", "degraded", "down"] = "ok"
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str = "0.5.0"
    components: dict[str, HealthComponent] = Field(default_factory=dict)


# Keep for backward compatibility
class AnalyzeResponse(BaseModel):
    """Deprecated — use TaskResponse + SessionResponse."""

    task_id: str
    status: str = "processing"
    result: CoachingFeedback | None = None
