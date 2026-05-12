"""Tool function: extract basketball shot frames from video.

Wraps the AI Shoot Phase 1 perception pipeline (VideoProcessor — pose
estimation + shot phase detection in one pass).

Uses deferred imports so the module loads in CI without heavy model weights.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_shot_frames(video_path: str) -> dict[str, Any]:
    """
    Extract key frames, pose keypoints, and shot phases from a basketball video.

    Runs the AI Shoot perception pipeline end-to-end via ``VideoProcessor``:
    1. Video decoding (OpenCV)
    2. Per-frame pose estimation (YOLOv11-pose or MediaPipe)
    3. Shot phase detection (setup / jump / release / follow-through)

    Args:
        video_path: Absolute or relative path to the video file.

    Returns:
        JSON-serializable dict matching ``PerceptionOutput`` schema:
        ``{video_path, fps, total_frames, player_detected, ball_detected,
        key_frames, shot_phases}``. On runtime failure, includes an
        ``error`` key so callers can surface a clean status to the user.
    """
    try:
        from src.perception.video_pipeline import VideoProcessor  # noqa: PLC0415

        processor = VideoProcessor()
        output = processor.process(video_path)
        return output.model_dump(mode="json")
    except ImportError as e:
        # Heavy CV deps (mediapipe / ultralytics / cv2) not installed in this
        # environment — legitimate CI-without-models case. Surface a stable
        # error code (not the raw module path) so the orchestrator can return
        # status=error and the user-facing message stays predictable.
        logger.error("Perception modules unavailable: %s", e)
        return {"error": "perception_unavailable", "video_path": video_path}
    except FileNotFoundError:
        # Path already lives in video_path — no need to repeat it in error.
        logger.error("Video file missing: %s", video_path)
        return {"error": "video_not_found", "video_path": video_path}
    except Exception as e:
        # Runtime failure inside the pipeline (corrupt video, CUDA OOM, etc.).
        # Full traceback goes to logs; the response carries only the exception
        # class name so we never leak internal state via the user-visible body.
        logger.exception("Perception failed for %s", video_path)
        return {"error": f"perception_failed:{type(e).__name__}", "video_path": video_path}
