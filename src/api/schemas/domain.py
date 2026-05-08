"""
Core data contracts for AI Shoot.

These schemas define the stable interfaces between all modules.
Implementation details can change — these shapes must not.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ShotResult(StrEnum):
    MADE = "made"
    MISSED = "missed"
    UNKNOWN = "unknown"


class ShotPhase(StrEnum):
    SETUP = "setup"
    JUMP = "jump"
    RELEASE = "release"
    FOLLOW_THROUGH = "follow_through"


class PlayerLevel(StrEnum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    ELITE = "elite"


class PoseModel(StrEnum):
    YOLOV11 = "yolov11-pose"
    VITPOSE = "vitpose"
    MEDIAPIPE = "mediapipe"


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class VideoInput(BaseModel):
    """Raw input to the analysis pipeline."""

    video_path: str = Field(..., description="Path to the video file")
    player_id: str | None = Field(None, description="Optional player identifier for session memory")
    player_level: PlayerLevel = Field(
        PlayerLevel.INTERMEDIATE, description="Self-reported skill level"
    )
    notes: str | None = Field(None, description="Optional coach/player notes for context")


class SessionContext(BaseModel):
    """Player context passed to the coaching agents."""

    player_id: str
    player_level: PlayerLevel
    session_count: int = Field(0, description="Number of previous sessions")
    recurring_issues: list[str] = Field(
        default_factory=list, description="Known recurring problems"
    )
    previous_drills: list[str] = Field(default_factory=list, description="Drills already assigned")
    notes: str | None = None


# ---------------------------------------------------------------------------
# Perception output
# ---------------------------------------------------------------------------


class Keypoint(BaseModel):
    """Single 2D/3D body keypoint."""

    name: str
    x: float
    y: float
    z: float | None = None
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


class PoseFrame(BaseModel):
    """Pose data for a single video frame."""

    frame_index: int
    timestamp_ms: float
    phase: ShotPhase | None = None
    keypoints: list[Keypoint]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


class PerceptionOutput(BaseModel):
    """Output from the perception module."""

    video_path: str
    fps: float
    total_frames: int
    player_detected: bool
    ball_detected: bool
    key_frames: list[PoseFrame] = Field(description="Sampled frames with pose data")
    shot_phases: dict[ShotPhase, int] = Field(
        default_factory=dict,
        description="Mapping phase → frame index where phase starts",
    )


# ---------------------------------------------------------------------------
# Analysis output (biomechanics)
# ---------------------------------------------------------------------------


class JointAngle(BaseModel):
    """Measured joint angle with reference range."""

    joint: str = Field(..., description="e.g. 'right_elbow', 'right_knee'")
    value_deg: float = Field(..., description="Measured angle in degrees")
    optimal_min: float | None = Field(None, description="Optimal range lower bound")
    optimal_max: float | None = Field(None, description="Optimal range upper bound")
    within_range: bool | None = None

    def model_post_init(self, __context: object) -> None:
        if self.optimal_min is not None and self.optimal_max is not None:
            self.within_range = self.optimal_min <= self.value_deg <= self.optimal_max


class ShotTimingMetrics(BaseModel):
    """Timing analysis of the shot phases."""

    setup_duration_ms: float | None = None
    jump_to_release_ms: float | None = None
    release_frame_index: int | None = None
    early_release: bool | None = None  # released before peak jump


class BallTrajectoryMetrics(BaseModel):
    """Ball arc and trajectory analysis."""

    launch_angle_deg: float | None = None
    optimal_launch_angle_min: float = 45.0
    optimal_launch_angle_max: float = 55.0
    peak_height_px: float | None = None
    trajectory_detected: bool = False


class BiomechanicsReport(BaseModel):
    """Full biomechanical analysis of a shot."""

    shot_result: ShotResult = ShotResult.UNKNOWN
    joint_angles: list[JointAngle] = Field(default_factory=list)
    timing: ShotTimingMetrics = Field(default_factory=ShotTimingMetrics)
    trajectory: BallTrajectoryMetrics = Field(default_factory=BallTrajectoryMetrics)
    alignment_score: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    primary_issue: str | None = Field(None, description="Single most important issue detected")
    issues_detected: list[str] = Field(default_factory=list, description="All detected issues")


# ---------------------------------------------------------------------------
# Coaching output
# ---------------------------------------------------------------------------


class Drill(BaseModel):
    """A recommended training drill."""

    name: str
    description: str
    duration_minutes: int
    focus: str = Field(..., description="What this drill specifically targets")
    difficulty: PlayerLevel = PlayerLevel.INTERMEDIATE


class CoachingFeedback(BaseModel):
    """
    Final output of the AI Shoot pipeline.

    This is what gets returned to the player/coach.
    """

    player_id: str | None = None
    shot_result: ShotResult
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]

    # Human-readable feedback (generated by VLM)
    summary: str = Field(..., description="2-3 sentence plain-language summary")
    primary_correction: str = Field(..., description="The ONE thing to work on now")
    detailed_analysis: str = Field(..., description="Full VLM-generated coaching text")

    # Structured data
    biomechanics: BiomechanicsReport
    drills: list[Drill] = Field(default_factory=list, max_length=3)

    # Metadata
    model_used: str = ""
    processing_time_ms: float | None = None
