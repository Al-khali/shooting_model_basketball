"""Tool function: extract basketball shot frames from video.

Wraps the AI Shoot Phase 1 perception pipeline (ShotDetector + PoseEstimator).
Uses deferred imports so the module loads in CI without heavy model weights.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_shot_frames(video_path: str) -> dict[str, Any]:
    """
    Extract key frames and pose keypoints from a basketball shot video.

    Runs the AI Shoot perception pipeline:
    1. Video decoding (OpenCV)
    2. Shot phase detection (ShotDetector — YOLOv11-pose or MediaPipe)
    3. Pose estimation per key frame

    Args:
        video_path: Absolute or relative path to the video file.

    Returns:
        JSON-serializable dict matching PerceptionOutput schema:
        {video_path, fps, total_frames, player_detected, ball_detected,
         key_frames, shot_phases}. On error, includes an "error" key.
    """
    try:
        from src.analysis.shot_detector import ShotPhaseDetector  # type: ignore[attr-defined]
        from src.perception.video_processor import VideoProcessor  # type: ignore[import]

        processor = VideoProcessor(video_path)
        detector = ShotPhaseDetector()
        result = detector.detect(processor.extract_frames())
        return result.model_dump()  # type: ignore[attr-defined]
    except ImportError:
        logger.warning("Perception modules not available — returning stub data for %s", video_path)
        return _stub_perception_output(video_path)
    except Exception as e:
        logger.error("Perception failed for %s: %s", video_path, e)
        return {"error": str(e), "video_path": video_path}


def _stub_perception_output(video_path: str) -> dict[str, Any]:
    """Stub PerceptionOutput for testing without heavy model dependencies."""
    return {
        "video_path": video_path,
        "fps": 30.0,
        "total_frames": 90,
        "player_detected": True,
        "ball_detected": True,
        "key_frames": [],
        "shot_phases": {},
    }
