"""
Video pipeline: video file → PerceptionOutput.

Orchestrates frame extraction, person detection, and pose estimation.
Designed for basketball shot analysis — focuses on shot sequences
rather than processing every frame (smart sampling).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.api.schemas.domain import (
    Keypoint,
    PerceptionOutput,
    PoseFrame,
    PoseModel,
    ShotPhase,
)
from src.perception.pose_estimator import BasePoseEstimator, PoseEstimatorFactory

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Internal config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Tuning parameters for the video pipeline."""

    # Pose estimation
    pose_model: PoseModel = PoseModel.YOLOV11
    device: str = "cpu"

    # Frame sampling
    max_frames_to_process: int = 150  # cap to limit inference cost
    sample_every_n_frames: int = 2    # process 1 frame every N (speed vs accuracy)
    min_keypoint_confidence: float = CONFIDENCE_THRESHOLD

    # Person detection heuristic — skip frames without enough valid keypoints
    min_keypoints_for_valid_pose: int = 10

    # Trajectory: frames around detected release to keep
    context_frames_before_release: int = 20
    context_frames_after_release: int = 15


# ---------------------------------------------------------------------------
# Frame sampler
# ---------------------------------------------------------------------------


def sample_frames(
    cap: cv2.VideoCapture,
    config: PipelineConfig,
) -> list[tuple[int, float, np.ndarray]]:
    """
    Extract sampled frames from a VideoCapture.

    Returns:
        List of (frame_index, timestamp_ms, frame_bgr) tuples.
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    sampled: list[tuple[int, float, np.ndarray]] = []
    frame_idx = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % config.sample_every_n_frames == 0:
            ts_ms = (frame_idx / fps) * 1000.0
            sampled.append((frame_idx, ts_ms, frame))
            processed += 1
            if processed >= config.max_frames_to_process:
                logger.debug("Reached max_frames_to_process=%d", config.max_frames_to_process)
                break

        frame_idx += 1

    logger.debug("Sampled %d/%d frames", len(sampled), total)
    return sampled


# ---------------------------------------------------------------------------
# Pose filter
# ---------------------------------------------------------------------------


def is_valid_pose(keypoints: list[Keypoint], min_valid: int) -> bool:
    """Return True if enough keypoints exceed the confidence threshold."""
    return sum(1 for kp in keypoints if kp.confidence >= CONFIDENCE_THRESHOLD) >= min_valid


# ---------------------------------------------------------------------------
# Main VideoProcessor
# ---------------------------------------------------------------------------


class VideoProcessor:
    """
    End-to-end processor: video file → PerceptionOutput.

    Usage::

        processor = VideoProcessor()
        output = processor.process("path/to/video.mp4")
    """

    def __init__(
        self,
        estimator: BasePoseEstimator | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._estimator = estimator or PoseEstimatorFactory.create(
            self._config.pose_model,
            device=self._config.device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, video_path: str | Path) -> PerceptionOutput:
        """
        Process a video file and return a PerceptionOutput.

        Args:
            video_path: Path to an mp4/mov/avi video file.

        Returns:
            PerceptionOutput with pose frames and shot phase annotations.

        Raises:
            FileNotFoundError: if the video file does not exist.
            ValueError: if the video cannot be opened by OpenCV.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"OpenCV cannot open video: {path}")

        try:
            return self._run(cap, str(path))
        finally:
            cap.release()

    def process_frames(
        self,
        frames: list[np.ndarray],
        fps: float = 30.0,
        source: str = "memory",
    ) -> PerceptionOutput:
        """
        Process a pre-loaded list of BGR frames (useful for testing / streaming).

        Args:
            frames: List of HxWxC BGR numpy arrays.
            fps: Frame rate to use for timestamp calculation.
            source: Label for the video_path field in the output.
        """
        sampled: list[tuple[int, float, np.ndarray]] = []
        for i, frame in enumerate(frames):
            if i % self._config.sample_every_n_frames == 0:
                sampled.append((i, (i / fps) * 1000.0, frame))
            if len(sampled) >= self._config.max_frames_to_process:
                break

        pose_frames = self._estimate_poses(sampled)
        shot_phases = self._detect_shot_phases(pose_frames)
        player_detected = len(pose_frames) > 0
        ball_detected = False  # object detection layer — future

        return PerceptionOutput(
            video_path=source,
            fps=fps,
            total_frames=len(frames),
            player_detected=player_detected,
            ball_detected=ball_detected,
            key_frames=pose_frames,
            shot_phases=shot_phases,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, cap: cv2.VideoCapture, video_path: str) -> PerceptionOutput:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        t0 = time.perf_counter()
        sampled = sample_frames(cap, self._config)

        pose_frames = self._estimate_poses(sampled)
        shot_phases = self._detect_shot_phases(pose_frames)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Processed %s: %d frames → %d pose frames in %.2fs",
            video_path,
            total_frames,
            len(pose_frames),
            elapsed,
        )

        return PerceptionOutput(
            video_path=video_path,
            fps=fps,
            total_frames=total_frames,
            player_detected=len(pose_frames) > 0,
            ball_detected=False,
            key_frames=pose_frames,
            shot_phases=shot_phases,
        )

    def _estimate_poses(
        self,
        sampled: list[tuple[int, float, np.ndarray]],
    ) -> list[PoseFrame]:
        pose_frames: list[PoseFrame] = []

        for frame_idx, ts_ms, frame in sampled:
            try:
                keypoints = self._estimator.estimate(frame)
            except Exception:
                logger.exception("Pose estimation failed on frame %d", frame_idx)
                continue

            if not is_valid_pose(keypoints, self._config.min_keypoints_for_valid_pose):
                continue

            # Per-frame confidence = mean of valid keypoints
            valid_confs = [kp.confidence for kp in keypoints if kp.confidence >= CONFIDENCE_THRESHOLD]
            frame_conf = float(np.mean(valid_confs)) if valid_confs else 0.0

            pose_frames.append(
                PoseFrame(
                    frame_index=frame_idx,
                    timestamp_ms=ts_ms,
                    keypoints=keypoints,
                    confidence=frame_conf,
                )
            )

        return pose_frames

    def _detect_shot_phases(
        self,
        pose_frames: list[PoseFrame],
    ) -> dict[ShotPhase, int]:
        """
        Lightweight heuristic phase detection based on hip vertical motion.

        Returns mapping of ShotPhase → frame_index where phase starts.
        Full shot phase detection is implemented in src/analysis/shot_detector.py;
        this quick version annotates PoseFrames in place.
        """
        if len(pose_frames) < 3:
            return {}

        # Import here to avoid circular imports
        from src.analysis.shot_detector import ShotPhaseDetector  # noqa: PLC0415

        detector = ShotPhaseDetector()
        return detector.detect(pose_frames)
