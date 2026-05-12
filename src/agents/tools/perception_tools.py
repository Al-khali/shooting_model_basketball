"""Tool function: extract basketball shot frames from video.

Wraps the AI Shoot Phase 1 perception pipeline (VideoProcessor — pose
estimation + shot phase detection in one pass).

Uses deferred imports so the module loads in CI without heavy model weights.
The ``VideoProcessor`` instance is cached at module level: the YOLO/MediaPipe
weights load from disk on the first ``extract_shot_frames`` call (~500ms–2s)
and are kept hot for subsequent calls. This is safe because ``VideoProcessor``
is stateless w.r.t. inputs — the underlying estimators are read-only after
``__init__``.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.perception.video_pipeline import VideoProcessor

logger = logging.getLogger(__name__)

# Lazy-initialized singleton — first call pays the model-load cost; later
# calls reuse the loaded weights. Guarded by a lock because FastAPI runs
# blocking tools in a worker pool via ``anyio.to_thread.run_sync``, so two
# concurrent /analyze requests can race here on a cold cache.
_processor: VideoProcessor | None = None
_processor_lock = threading.Lock()


def _get_processor() -> VideoProcessor:
    """Return the cached ``VideoProcessor``, building it on first call."""
    global _processor
    if _processor is None:
        with _processor_lock:
            if _processor is None:
                from src.perception.video_pipeline import (  # noqa: PLC0415
                    VideoProcessor as _VP,
                )

                _processor = _VP()
                logger.info("VideoProcessor initialized — models loaded into memory")
    return _processor


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
        processor = _get_processor()
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
