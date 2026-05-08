"""API request/response schemas (HTTP layer)."""

from pydantic import BaseModel, Field

from src.api.schemas.domain import CoachingFeedback, PlayerLevel


class AnalyzeResponse(BaseModel):
    """Response from POST /analyze."""

    task_id: str
    status: str = "processing"
    result: CoachingFeedback | None = None


class SessionHistoryItem(BaseModel):
    task_id: str
    shot_result: str
    primary_correction: str
    timestamp: str


class PlayerHistoryResponse(BaseModel):
    player_id: str
    player_level: PlayerLevel
    session_count: int
    recurring_issues: list[str]
    recent_sessions: list[SessionHistoryItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
