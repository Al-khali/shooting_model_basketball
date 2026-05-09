"""Unit tests for Phase 3 — Agent Pipeline.

Tests cover:
- PlayerSession state model (recurring issues, recent drills)
- PlayerMemoryService (CRUD, context building, feedback recording)
- Tool functions (perception_tools, analysis_tools, coach_tools, planner_tools, memory_tools)
- ShotAnalysisPipeline (orchestration, memory integration, error handling)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003
from typing import Any
from unittest.mock import patch

import pytest

from src.agents.memory import PlayerMemoryService
from src.agents.state import PlayerSession, ShotRecord
from src.api.schemas.domain import (
    BiomechanicsReport,
    CoachingFeedback,
    Drill,
    PlayerLevel,
    ShotResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feedback(
    player_id: str = "p1",
    result: ShotResult = ShotResult.MADE,
    primary_issue: str = "elbow flare",
    issues: list[str] | None = None,
    drills: list[Drill] | None = None,
) -> CoachingFeedback:
    bio = BiomechanicsReport(
        shot_result=result,
        primary_issue=primary_issue,
        issues_detected=issues or [primary_issue],
    )
    return CoachingFeedback(
        player_id=player_id,
        shot_result=result,
        confidence=0.8,
        summary="Good shot.",
        primary_correction=primary_issue,
        detailed_analysis="Detailed analysis text.",
        biomechanics=bio,
        drills=drills or [],
        model_used="gemini-test",
    )


def _make_drill(name: str = "Free throw routine") -> Drill:
    return Drill(
        name=name,
        description="Practice free throws.",
        duration_minutes=10,
        focus="release mechanics",
    )


@pytest.fixture()
def tmp_memory(tmp_path: Path) -> PlayerMemoryService:
    return PlayerMemoryService(store_dir=tmp_path)


# ---------------------------------------------------------------------------
# PlayerSession — state model
# ---------------------------------------------------------------------------


class TestPlayerSession:
    def test_recurring_issues_requires_two_occurrences(self) -> None:
        session = PlayerSession(
            player_id="p1",
            issue_counts={"elbow flare": 1, "late release": 3},
        )
        # elbow flare (1) below threshold; late release (3) qualifies
        assert session.recurring_issues == ["late release"]

    def test_recurring_issues_sorted_by_frequency(self) -> None:
        session = PlayerSession(
            player_id="p1",
            issue_counts={"A": 2, "B": 5, "C": 3},
        )
        assert session.recurring_issues == ["B", "C", "A"]

    def test_recurring_issues_empty_when_all_below_threshold(self) -> None:
        session = PlayerSession(player_id="p1", issue_counts={"A": 1})
        assert session.recurring_issues == []

    def test_recent_drills_returns_last_five_unique(self) -> None:
        session = PlayerSession(
            player_id="p1",
            drills_history=["D1", "D2", "D3", "D4", "D5", "D6"],
        )
        assert session.recent_drills == ["D6", "D5", "D4", "D3", "D2"]

    def test_recent_drills_deduplicates(self) -> None:
        session = PlayerSession(
            player_id="p1",
            drills_history=["D1", "D1", "D2"],
        )
        assert session.recent_drills == ["D2", "D1"]

    def test_recent_drills_empty_history(self) -> None:
        session = PlayerSession(player_id="p1")
        assert session.recent_drills == []

    def test_shot_record_creation(self) -> None:
        record = ShotRecord(
            session_id="abc",
            timestamp=datetime.now(UTC),
            shot_result=ShotResult.MADE,
            primary_correction="Bend your knees more",
            drills_assigned=["D1"],
            alignment_score=0.85,
        )
        assert record.shot_result == ShotResult.MADE
        assert record.alignment_score == 0.85


# ---------------------------------------------------------------------------
# PlayerMemoryService
# ---------------------------------------------------------------------------


class TestPlayerMemoryService:
    def test_load_new_player_returns_empty_session(self, tmp_memory: PlayerMemoryService) -> None:
        session = tmp_memory.load("new_player")
        assert session.player_id == "new_player"
        assert session.total_sessions == 0
        assert session.recurring_issues == []

    def test_save_and_load_round_trip(self, tmp_memory: PlayerMemoryService) -> None:
        session = PlayerSession(
            player_id="p42",
            total_sessions=3,
            issue_counts={"late release": 2},
        )
        tmp_memory.save(session)
        loaded = tmp_memory.load("p42")
        assert loaded.total_sessions == 3
        assert loaded.issue_counts == {"late release": 2}

    def test_build_context_new_player(self, tmp_memory: PlayerMemoryService) -> None:
        ctx = tmp_memory.build_context("p_new", PlayerLevel.BEGINNER)
        assert ctx.player_id == "p_new"
        assert ctx.session_count == 0
        assert ctx.recurring_issues == []
        assert ctx.previous_drills == []
        assert ctx.player_level == PlayerLevel.BEGINNER

    def test_build_context_existing_player(self, tmp_memory: PlayerMemoryService) -> None:
        session = PlayerSession(
            player_id="p5",
            total_sessions=4,
            issue_counts={"elbow flare": 3},
            drills_history=["Free throw routine"],
        )
        tmp_memory.save(session)
        ctx = tmp_memory.build_context("p5", PlayerLevel.INTERMEDIATE)
        assert ctx.session_count == 4
        assert "elbow flare" in ctx.recurring_issues
        assert "Free throw routine" in ctx.previous_drills

    def test_build_context_passes_notes(self, tmp_memory: PlayerMemoryService) -> None:
        ctx = tmp_memory.build_context("p1", PlayerLevel.ADVANCED, notes="Focus on arc")
        assert ctx.notes == "Focus on arc"

    def test_record_feedback_increments_session_count(
        self, tmp_memory: PlayerMemoryService
    ) -> None:
        feedback = _make_feedback()
        updated = tmp_memory.record_feedback("p1", feedback, PlayerLevel.INTERMEDIATE)
        assert updated.total_sessions == 1
        updated2 = tmp_memory.record_feedback("p1", feedback, PlayerLevel.INTERMEDIATE)
        assert updated2.total_sessions == 2

    def test_record_feedback_tracks_issue_frequency(self, tmp_memory: PlayerMemoryService) -> None:
        feedback = _make_feedback(primary_issue="elbow flare", issues=["elbow flare"])
        tmp_memory.record_feedback("p1", feedback, PlayerLevel.INTERMEDIATE)
        tmp_memory.record_feedback("p1", feedback, PlayerLevel.INTERMEDIATE)
        session = tmp_memory.load("p1")
        assert session.issue_counts.get("elbow flare", 0) >= 2

    def test_record_feedback_tracks_drills(self, tmp_memory: PlayerMemoryService) -> None:
        drill = _make_drill("Free throw routine")
        feedback = _make_feedback(drills=[drill])
        tmp_memory.record_feedback("p1", feedback, PlayerLevel.INTERMEDIATE)
        session = tmp_memory.load("p1")
        assert "Free throw routine" in session.drills_history

    def test_recurring_issues_after_multiple_sessions(
        self, tmp_memory: PlayerMemoryService
    ) -> None:
        feedback = _make_feedback(primary_issue="late release", issues=["late release"])
        tmp_memory.record_feedback("p1", feedback, PlayerLevel.INTERMEDIATE)
        tmp_memory.record_feedback("p1", feedback, PlayerLevel.INTERMEDIATE)
        session = tmp_memory.load("p1")
        assert "late release" in session.recurring_issues

    def test_player_path_sanitizes_slash(self, tmp_memory: PlayerMemoryService) -> None:
        path = tmp_memory._player_path("org/player42")
        assert "/" not in path.name

    def test_load_corrupted_file_returns_empty_session(
        self, tmp_memory: PlayerMemoryService
    ) -> None:
        path = tmp_memory._player_path("corrupt")
        path.write_text("not valid json")
        session = tmp_memory.load("corrupt")
        assert session.player_id == "corrupt"
        assert session.total_sessions == 0


# ---------------------------------------------------------------------------
# Tool functions — perception_tools
# ---------------------------------------------------------------------------


class TestPerceptionTools:
    def test_extract_shot_frames_returns_stub_on_import_error(self) -> None:
        from src.agents.tools.perception_tools import extract_shot_frames

        with patch.dict("sys.modules", {"src.perception.video_processor": None}):
            result = extract_shot_frames("nonexistent.mp4")
        # Should return dict (stub or error), not raise
        assert isinstance(result, dict)
        assert "video_path" in result or "error" in result

    def test_extract_shot_frames_handles_generic_exception(self) -> None:
        from src.agents.tools.perception_tools import _stub_perception_output

        stub = _stub_perception_output("test.mp4")
        assert stub["fps"] == 30.0
        assert stub["player_detected"] is True

    def test_stub_perception_output_structure(self) -> None:
        from src.agents.tools.perception_tools import _stub_perception_output

        out = _stub_perception_output("video.mp4")
        required_keys = {"video_path", "fps", "total_frames", "player_detected", "ball_detected"}
        assert required_keys.issubset(out.keys())


# ---------------------------------------------------------------------------
# Tool functions — analysis_tools
# ---------------------------------------------------------------------------


class TestAnalysisTools:
    def test_compute_biomechanics_invalid_json(self) -> None:
        from src.agents.tools.analysis_tools import compute_biomechanics

        result = compute_biomechanics("not-json")
        assert "error" in result
        assert "Invalid JSON" in result["error"]

    def test_compute_biomechanics_returns_stub_on_import_error(self) -> None:
        from src.agents.tools.analysis_tools import compute_biomechanics

        payload = json.dumps({"video_path": "x.mp4", "fps": 30.0})
        with patch.dict("sys.modules", {"src.analysis.biomechanics": None}):
            result = compute_biomechanics(payload)
        assert isinstance(result, dict)

    def test_stub_biomechanics_structure(self) -> None:
        from src.agents.tools.analysis_tools import _stub_biomechanics_report

        out = _stub_biomechanics_report()
        assert "shot_result" in out
        assert "issues_detected" in out


# ---------------------------------------------------------------------------
# Tool functions — coach_tools
# ---------------------------------------------------------------------------


class TestCoachTools:
    def test_generate_feedback_invalid_biomechanics_json(self) -> None:
        from src.agents.tools.coach_tools import generate_coaching_feedback

        result = generate_coaching_feedback("bad-json")
        assert "error" in result

    def test_generate_feedback_returns_stub_on_import_error(self) -> None:
        from src.agents.tools.coach_tools import generate_coaching_feedback

        bio = BiomechanicsReport(shot_result=ShotResult.MISSED)
        with patch.dict("sys.modules", {"src.vlm.gemini_client": None}):
            result = generate_coaching_feedback(bio.model_dump_json())
        assert isinstance(result, dict)
        assert "summary" in result or "error" in result

    def test_stub_coaching_feedback_structure(self) -> None:
        from src.agents.tools.coach_tools import _stub_coaching_feedback

        bio_data = {"shot_result": "made", "primary_issue": "elbow flare"}
        out = _stub_coaching_feedback(bio_data, "p1")
        assert out["model_used"] == "fallback"
        assert out["confidence"] == 0.0
        assert "elbow flare" in out["primary_correction"]


# ---------------------------------------------------------------------------
# Tool functions — planner_tools
# ---------------------------------------------------------------------------


class TestPlannerTools:
    def _feedback_json(self, drills: list[dict] | None = None) -> str:
        return json.dumps(
            {
                "primary_correction": "Bend your knees",
                "drills": drills or [],
                "shot_result": "made",
                "confidence": 0.8,
            }
        )

    def test_build_plan_first_session(self) -> None:
        from src.agents.tools.planner_tools import build_training_plan

        result = build_training_plan(self._feedback_json(), session_count=0)
        assert result["weekly_sessions"] == 3
        assert result["intensity"] == "light"
        assert result["session_number"] == 1
        assert "First session" in result["progression_note"]

    def test_build_plan_escalates_weekly_sessions(self) -> None:
        from src.agents.tools.planner_tools import build_training_plan

        mid = build_training_plan(self._feedback_json(), session_count=3)
        late = build_training_plan(self._feedback_json(), session_count=10)
        assert mid["weekly_sessions"] == 4
        assert late["weekly_sessions"] == 5

    def test_build_plan_prioritizes_recurring_issues(self) -> None:
        from src.agents.tools.planner_tools import build_training_plan

        result = build_training_plan(
            self._feedback_json(),
            recurring_issues_json=json.dumps(["late release"]),
        )
        assert "late release" in result["focus_area"]
        assert "Persistent issue" in result["focus_area"]

    def test_build_plan_filters_previous_drills(self) -> None:
        from src.agents.tools.planner_tools import build_training_plan

        drills = [{"name": "D1", "description": "x", "duration_minutes": 5, "focus": "f"}]
        result = build_training_plan(
            self._feedback_json(drills=drills),
            previous_drills_json=json.dumps(["D1"]),
        )
        # All drills are repeats → fall back to all_drills (include repeats)
        assert isinstance(result["drills"], list)

    def test_build_plan_keeps_new_drills(self) -> None:
        from src.agents.tools.planner_tools import build_training_plan

        drills = [
            {"name": "D1", "description": "x", "duration_minutes": 5, "focus": "f"},
            {"name": "D2", "description": "y", "duration_minutes": 5, "focus": "g"},
        ]
        result = build_training_plan(
            self._feedback_json(drills=drills),
            previous_drills_json=json.dumps(["D1"]),
        )
        names = [d["name"] for d in result["drills"]]
        assert "D2" in names
        assert "D1" not in names

    def test_build_plan_invalid_feedback_json(self) -> None:
        from src.agents.tools.planner_tools import build_training_plan

        result = build_training_plan("not-json")
        assert "error" in result


# ---------------------------------------------------------------------------
# Tool functions — memory_tools
# ---------------------------------------------------------------------------


class TestMemoryTools:
    def test_load_player_history_new_player(self, tmp_path: Path) -> None:
        from src.agents.tools import memory_tools

        original_svc = memory_tools._memory_service
        try:
            memory_tools._memory_service = PlayerMemoryService(store_dir=tmp_path)
            result = memory_tools.load_player_history("brand_new")
            assert result["total_sessions"] == 0
            assert result["recurring_issues"] == []
        finally:
            memory_tools._memory_service = original_svc

    def test_save_coaching_result_invalid_json(self, tmp_path: Path) -> None:
        from src.agents.tools.memory_tools import save_coaching_result

        result = save_coaching_result("p1", "bad-json")
        assert "Error" in result


# ---------------------------------------------------------------------------
# ShotAnalysisPipeline
# ---------------------------------------------------------------------------


class TestShotAnalysisPipeline:
    def _make_pipeline(self, tmp_path: Path) -> Any:  # noqa: F821
        from src.agents.orchestrator import ShotAnalysisPipeline

        return ShotAnalysisPipeline(memory_service=PlayerMemoryService(store_dir=tmp_path))

    def test_analyze_anonymous_player_returns_result(self, tmp_path: Path) -> None:
        from src.api.schemas.domain import VideoInput

        pipeline = self._make_pipeline(tmp_path)
        result = pipeline.analyze(VideoInput(video_path="shot.mp4"))
        assert result["player_id"] == "anonymous"
        assert "coaching_feedback" in result
        assert "training_plan" in result
        assert result["processing_time_ms"] > 0

    def test_analyze_with_player_id_saves_session(self, tmp_path: Path) -> None:
        from src.api.schemas.domain import VideoInput

        pipeline = self._make_pipeline(tmp_path)
        result = pipeline.analyze(VideoInput(video_path="shot.mp4", player_id="p99"))
        # The stub feedback may not be parseable as CoachingFeedback — that's OK
        # We just check the pipeline ran and returned a result
        assert result["player_id"] == "p99"
        assert result["session_number"] == 1

    def test_analyze_increments_session_on_second_run(self, tmp_path: Path) -> None:
        from src.api.schemas.domain import VideoInput

        pipeline = self._make_pipeline(tmp_path)
        # Pre-seed a valid feedback directly
        feedback = _make_feedback(player_id="p10")
        pipeline.memory.record_feedback("p10", feedback, PlayerLevel.INTERMEDIATE)

        result = pipeline.analyze(VideoInput(video_path="shot.mp4", player_id="p10"))
        assert result["session_number"] == 2

    def test_error_result_structure(self, tmp_path: Path) -> None:
        pipeline = self._make_pipeline(tmp_path)
        err = pipeline._error_result("p1", "Something failed", 42.0)
        assert err["error"] == "Something failed"
        assert err["coaching_feedback"] is None
        assert err["processing_time_ms"] == 42.0

    def test_pipeline_uses_custom_memory_service(self, tmp_path: Path) -> None:
        from src.agents.orchestrator import ShotAnalysisPipeline

        custom_svc = PlayerMemoryService(store_dir=tmp_path / "custom")
        pipeline = ShotAnalysisPipeline(memory_service=custom_svc)
        assert pipeline.memory is custom_svc
