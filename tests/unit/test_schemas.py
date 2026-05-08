"""Tests on core Pydantic schemas — the stable contracts of the project."""

import pytest
from pydantic import ValidationError

from src.api.schemas.domain import (
    BiomechanicsReport,
    CoachingFeedback,
    JointAngle,
    Keypoint,
    PlayerLevel,
    SessionContext,
    ShotResult,
    VideoInput,
)


class TestVideoInput:
    def test_minimal(self):
        v = VideoInput(video_path="shot.mp4")
        assert v.video_path == "shot.mp4"
        assert v.player_level == PlayerLevel.INTERMEDIATE
        assert v.player_id is None

    def test_full(self):
        v = VideoInput(
            video_path="shot.mp4",
            player_id="player_42",
            player_level=PlayerLevel.ADVANCED,
            notes="Working on release point",
        )
        assert v.player_id == "player_42"
        assert v.player_level == PlayerLevel.ADVANCED

    def test_requires_video_path(self):
        with pytest.raises(ValidationError):
            VideoInput()


class TestJointAngle:
    def test_within_range_computed_true(self):
        j = JointAngle(joint="right_elbow", value_deg=33.0, optimal_min=30.0, optimal_max=35.0)
        assert j.within_range is True

    def test_within_range_computed_false(self):
        j = JointAngle(joint="right_elbow", value_deg=52.0, optimal_min=30.0, optimal_max=35.0)
        assert j.within_range is False

    def test_no_range_no_flag(self):
        j = JointAngle(joint="right_elbow", value_deg=33.0)
        assert j.within_range is None


class TestKeypoint:
    def test_2d_keypoint(self):
        k = Keypoint(name="right_elbow", x=0.5, y=0.3)
        assert k.z is None
        assert k.confidence == 1.0

    def test_3d_keypoint(self):
        k = Keypoint(name="right_elbow", x=0.5, y=0.3, z=0.1, confidence=0.92)
        assert k.z == 0.1
        assert k.confidence == 0.92


class TestCoachingFeedback:
    def test_valid_feedback(self):
        feedback = CoachingFeedback(
            shot_result=ShotResult.MISSED,
            confidence=0.85,
            summary="Ton coude sort trop loin du corps au moment du release.",
            primary_correction="Garde le coude à 30-35° pendant le jump.",
            detailed_analysis="Analyse complète...",
            biomechanics=BiomechanicsReport(
                shot_result=ShotResult.MISSED,
                primary_issue="Elbow angle too wide",
            ),
        )
        assert feedback.shot_result == ShotResult.MISSED
        assert feedback.confidence == 0.85
        assert len(feedback.drills) == 0

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            CoachingFeedback(
                shot_result=ShotResult.MADE,
                confidence=1.5,  # invalid
                summary="x",
                primary_correction="x",
                detailed_analysis="x",
                biomechanics=BiomechanicsReport(),
            )


class TestSessionContext:
    def test_defaults(self):
        ctx = SessionContext(player_id="p1", player_level=PlayerLevel.BEGINNER)
        assert ctx.session_count == 0
        assert ctx.recurring_issues == []
        assert ctx.previous_drills == []
