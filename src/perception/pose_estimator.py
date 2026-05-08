"""
Pose estimation abstraction layer.

Supports YOLOv11-pose (default), MediaPipe BlazePose, and ViTPose (ONNX).
All estimators return a unified list[Keypoint] using COCO-17 body keypoints.
Heavy model imports are deferred to __init__ to keep import time low.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from src.api.schemas.domain import Keypoint, PoseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO-17 keypoint index → name mapping (used by YOLO and ViTPose)
# ---------------------------------------------------------------------------

COCO_17_KEYPOINTS: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# MediaPipe BlazePose 33-keypoint subset we care about → COCO-17 equivalent names
# Only the 17 joints that matter for basketball shooting mechanics
MEDIAPIPE_TO_COCO: dict[int, str] = {
    0: "nose",
    2: "left_eye",
    5: "right_eye",
    7: "left_ear",
    8: "right_ear",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

CONFIDENCE_THRESHOLD = 0.3  # below this, keypoint is considered missing


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BasePoseEstimator(ABC):
    """Abstract base for all pose estimators."""

    @abstractmethod
    def estimate(self, frame: np.ndarray) -> list[Keypoint]:
        """
        Run pose estimation on a single BGR frame.

        Args:
            frame: HxWxC numpy array in BGR format (OpenCV standard).

        Returns:
            List of Keypoints. May be empty if no person detected.
            Keypoints below CONFIDENCE_THRESHOLD are included but marked low-confidence.
        """

    @abstractmethod
    def model_name(self) -> PoseModel:
        """Return the PoseModel enum value for this estimator."""

    def warmup(self) -> None:
        """Optional: run a dummy inference to pre-load CUDA kernels etc."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.estimate(dummy)
        logger.debug("Warmup complete for %s", self.model_name())


# ---------------------------------------------------------------------------
# YOLOv11-pose estimator
# ---------------------------------------------------------------------------


class YOLOPoseEstimator(BasePoseEstimator):
    """
    YOLOv11-pose estimator via the ultralytics library.

    Best for: real-time inference (30+ fps), single/multi-person scenes,
    CPU and MPS (Apple Silicon) friendly. 17 COCO keypoints.

    Model weights are downloaded automatically on first use.
    """

    def __init__(self, model_path: str = "yolo11n-pose.pt", device: str = "cpu") -> None:
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("ultralytics is required: uv add ultralytics") from e

        logger.info("Loading YOLO pose model: %s on %s", model_path, device)
        self._model = YOLO(model_path)
        self._device = device

    def estimate(self, frame: np.ndarray) -> list[Keypoint]:
        results = self._model(frame, device=self._device, verbose=False)
        if not results or results[0].keypoints is None:
            return []

        kp_data = results[0].keypoints.data
        if kp_data.shape[0] == 0:
            return []

        # Take the highest-confidence person (index 0 after YOLO sorts by conf)
        person_kps = kp_data[0].cpu().numpy()  # shape: (17, 3) → x, y, conf

        keypoints: list[Keypoint] = []
        for idx, (x, y, conf) in enumerate(person_kps):
            if idx >= len(COCO_17_KEYPOINTS):
                break
            keypoints.append(
                Keypoint(
                    name=COCO_17_KEYPOINTS[idx],
                    x=float(x),
                    y=float(y),
                    confidence=float(conf),
                )
            )
        return keypoints

    def model_name(self) -> PoseModel:
        return PoseModel.YOLOV11


# ---------------------------------------------------------------------------
# MediaPipe BlazePose estimator
# ---------------------------------------------------------------------------


class MediaPipePoseEstimator(BasePoseEstimator):
    """
    MediaPipe BlazePose estimator.

    Best for: 3D keypoints, full-body with hands/face, edge devices.
    Returns z-coordinate (depth relative to hips). 33 landmarks total,
    mapped to the COCO-17 subset for unified downstream use.
    """

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        try:
            import mediapipe as mp  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("mediapipe is required: uv add mediapipe") from e

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
        )

    def estimate(self, frame: np.ndarray) -> list[Keypoint]:
        import cv2  # noqa: PLC0415

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return []

        h, w = frame.shape[:2]
        keypoints: list[Keypoint] = []
        landmarks = results.pose_landmarks.landmark

        for mp_idx, coco_name in MEDIAPIPE_TO_COCO.items():
            lm = landmarks[mp_idx]
            keypoints.append(
                Keypoint(
                    name=coco_name,
                    x=lm.x * w,
                    y=lm.y * h,
                    z=lm.z,  # normalized depth
                    confidence=lm.visibility,
                )
            )
        return keypoints

    def model_name(self) -> PoseModel:
        return PoseModel.MEDIAPIPE

    def __del__(self) -> None:
        if hasattr(self, "_pose"):
            self._pose.close()


# ---------------------------------------------------------------------------
# ViTPose stub (ONNX — Phase 1 benchmark candidate, full impl in Phase 2)
# ---------------------------------------------------------------------------


class ViTPoseEstimator(BasePoseEstimator):
    """
    ViTPose via ONNX runtime.

    State-of-the-art accuracy (133 keypoints with WholeBody,
    or 17 with Body-only). Requires a 2-stage pipeline:
    1. Person detector (YOLO or RT-DETR)
    2. ViTPose on each detected person crop

    Status: stub — raises NotImplementedError until Phase 2.
    Tracking issue: https://github.com/Al-khali/shooting_model_basketball/issues/10
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "ViTPose is not yet implemented. "
            "Use YOLOPoseEstimator or MediaPipePoseEstimator for now. "
            "Track progress: https://github.com/Al-khali/shooting_model_basketball/issues/10"
        )

    def estimate(self, frame: np.ndarray) -> list[Keypoint]:  # pragma: no cover
        raise NotImplementedError

    def model_name(self) -> PoseModel:  # pragma: no cover
        return PoseModel.VITPOSE


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class PoseEstimatorFactory:
    """Create the right estimator based on config."""

    @staticmethod
    def create(model: PoseModel | str, device: str = "cpu") -> BasePoseEstimator:
        """
        Instantiate a pose estimator.

        Args:
            model: PoseModel enum value or string ("yolov11-pose", "mediapipe", "vitpose").
            device: "cpu", "cuda", or "mps".

        Returns:
            Configured BasePoseEstimator instance.
        """
        if isinstance(model, str):
            model = PoseModel(model)

        match model:
            case PoseModel.YOLOV11:
                return YOLOPoseEstimator(device=device)
            case PoseModel.MEDIAPIPE:
                return MediaPipePoseEstimator()
            case PoseModel.VITPOSE:
                raise NotImplementedError(
                    "ViTPose is not yet available. See issue #10 for tracking."
                )
            case _:
                raise ValueError(f"Unknown pose model: {model}")
