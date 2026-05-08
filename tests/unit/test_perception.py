"""
Tests for src/perception/pose_estimator.py and src/perception/video_pipeline.py

Uses mocked estimators — no real model weights needed.
Heavy model imports (ultralytics, mediapipe) are tested via mock to keep CI fast.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.api.schemas.domain import Keypoint, PerceptionOutput, PoseModel, ShotPhase
from src.perception.pose_estimator import (
    COCO_17_KEYPOINTS,
    BasePoseEstimator,
    PoseEstimatorFactory,
    ViTPoseEstimator,
)
from src.perception.video_pipeline import (
    PipelineConfig,
    VideoProcessor,
    is_valid_pose,
    sample_frames,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def make_keypoint(name: str, conf: float = 0.9) -> Keypoint:
    return Keypoint(name=name, x=100.0, y=200.0, confidence=conf)


def full_coco17_keypoints(conf: float = 0.9) -> list[Keypoint]:
    return [make_keypoint(name, conf) for name in COCO_17_KEYPOINTS]


class MockEstimator(BasePoseEstimator):
    """Deterministic mock estimator that returns synthetic COCO-17 keypoints."""

    def __init__(self, keypoints_per_frame: list[Keypoint] | None = None) -> None:
        self._keypoints = keypoints_per_frame or full_coco17_keypoints()

    def estimate(self, frame: np.ndarray) -> list[Keypoint]:  # noqa: ARG002
        return self._keypoints

    def model_name(self) -> PoseModel:
        return PoseModel.YOLOV11


class EmptyEstimator(BasePoseEstimator):
    """Always returns no keypoints (no person detected)."""

    def estimate(self, frame: np.ndarray) -> list[Keypoint]:  # noqa: ARG002
        return []

    def model_name(self) -> PoseModel:
        return PoseModel.YOLOV11


def make_bgr_frames(n: int = 10, h: int = 48, w: int = 64) -> list[np.ndarray]:
    """Generate n random BGR frames."""
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


# ---------------------------------------------------------------------------
# PoseEstimatorFactory
# ---------------------------------------------------------------------------


class TestPoseEstimatorFactory:
    def test_vitpose_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            PoseEstimatorFactory.create(PoseModel.VITPOSE)

    def test_unknown_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            PoseEstimatorFactory.create("unknown_model")  # type: ignore[arg-type]

    def test_yolo_created_when_ultralytics_available(self) -> None:
        mock_yolo = MagicMock()
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo)}):
            estimator = PoseEstimatorFactory.create(PoseModel.YOLOV11)
            assert estimator.model_name() == PoseModel.YOLOV11

    def test_mediapipe_created_when_available(self) -> None:
        mock_mp = MagicMock()
        mock_mp.solutions.pose.Pose.return_value = MagicMock()
        with patch.dict("sys.modules", {"mediapipe": mock_mp}):
            estimator = PoseEstimatorFactory.create(PoseModel.MEDIAPIPE)
            assert estimator.model_name() == PoseModel.MEDIAPIPE


# ---------------------------------------------------------------------------
# ViTPoseEstimator (stub)
# ---------------------------------------------------------------------------


class TestViTPoseEstimator:
    def test_instantiation_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="ViTPose is not yet implemented"):
            ViTPoseEstimator()


# ---------------------------------------------------------------------------
# is_valid_pose
# ---------------------------------------------------------------------------


class TestIsValidPose:
    def test_enough_confident_keypoints(self) -> None:
        kps = [make_keypoint("k", conf=0.9) for _ in range(12)]
        assert is_valid_pose(kps, min_valid=10) is True

    def test_not_enough_confident_keypoints(self) -> None:
        kps = [make_keypoint("k", conf=0.9) for _ in range(5)]
        assert is_valid_pose(kps, min_valid=10) is False

    def test_all_low_confidence(self) -> None:
        kps = [make_keypoint("k", conf=0.1) for _ in range(17)]
        assert is_valid_pose(kps, min_valid=5) is False

    def test_empty_list(self) -> None:
        assert is_valid_pose([], min_valid=1) is False


# ---------------------------------------------------------------------------
# VideoProcessor.process_frames
# ---------------------------------------------------------------------------


class TestVideoProcessorProcessFrames:
    def test_returns_perception_output(self) -> None:
        frames = make_bgr_frames(30)
        processor = VideoProcessor(estimator=MockEstimator())
        output = processor.process_frames(frames, fps=30.0)
        assert isinstance(output, PerceptionOutput)

    def test_player_detected_when_keypoints_present(self) -> None:
        frames = make_bgr_frames(20)
        processor = VideoProcessor(estimator=MockEstimator())
        output = processor.process_frames(frames, fps=30.0)
        assert output.player_detected is True

    def test_player_not_detected_when_empty_estimator(self) -> None:
        frames = make_bgr_frames(20)
        processor = VideoProcessor(estimator=EmptyEstimator())
        output = processor.process_frames(frames, fps=30.0)
        assert output.player_detected is False

    def test_fps_stored_correctly(self) -> None:
        frames = make_bgr_frames(10)
        processor = VideoProcessor(estimator=MockEstimator())
        output = processor.process_frames(frames, fps=60.0)
        assert output.fps == 60.0

    def test_total_frames_is_input_length(self) -> None:
        frames = make_bgr_frames(25)
        processor = VideoProcessor(estimator=MockEstimator())
        output = processor.process_frames(frames)
        assert output.total_frames == 25

    def test_source_label_in_output(self) -> None:
        frames = make_bgr_frames(5)
        processor = VideoProcessor(estimator=MockEstimator())
        output = processor.process_frames(frames, source="test_video.mp4")
        assert output.video_path == "test_video.mp4"

    def test_empty_frames_returns_no_player(self) -> None:
        processor = VideoProcessor(estimator=MockEstimator())
        output = processor.process_frames([], fps=30.0)
        assert output.player_detected is False
        assert output.key_frames == []

    def test_shot_phases_dict_is_dict(self) -> None:
        frames = make_bgr_frames(60)
        processor = VideoProcessor(estimator=MockEstimator())
        output = processor.process_frames(frames, fps=30.0)
        assert isinstance(output.shot_phases, dict)

    def test_timestamps_increase_monotonically(self) -> None:
        frames = make_bgr_frames(60)
        processor = VideoProcessor(estimator=MockEstimator())
        output = processor.process_frames(frames, fps=30.0)
        if len(output.key_frames) > 1:
            timestamps = [f.timestamp_ms for f in output.key_frames]
            assert all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))


# ---------------------------------------------------------------------------
# VideoProcessor.process (file-based)
# ---------------------------------------------------------------------------


class TestVideoProcessorProcessFile:
    def test_file_not_found_raises(self) -> None:
        processor = VideoProcessor(estimator=MockEstimator())
        with pytest.raises(FileNotFoundError):
            processor.process("/nonexistent/path/video.mp4")

    def test_invalid_file_raises_value_error(self, tmp_path: pytest.TempPathFactory) -> None:
        invalid = tmp_path / "invalid.mp4"
        invalid.write_bytes(b"not a video")
        processor = VideoProcessor(estimator=MockEstimator())
        with pytest.raises(ValueError, match="OpenCV cannot open"):
            processor.process(invalid)


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_defaults(self) -> None:
        config = PipelineConfig()
        assert config.pose_model == PoseModel.YOLOV11
        assert config.device == "cpu"
        assert config.max_frames_to_process == 150
        assert config.sample_every_n_frames == 2
        assert config.min_keypoint_confidence == 0.3

    def test_custom_config(self) -> None:
        config = PipelineConfig(
            pose_model=PoseModel.MEDIAPIPE,
            device="mps",
            max_frames_to_process=50,
        )
        assert config.pose_model == PoseModel.MEDIAPIPE
        assert config.device == "mps"
        assert config.max_frames_to_process == 50
