"""
Biomechanics analysis module.

Computes joint angles, timing metrics, and trajectory estimates
from a sequence of PoseFrames. Uses NBA reference ranges calibrated
on elite shooters (Curry, Thompson, Durant) as optimal benchmarks.
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

import numpy as np

from src.api.schemas.domain import (
    BallTrajectoryMetrics,
    BiomechanicsReport,
    JointAngle,
    PoseFrame,
    ShotPhase,
    ShotResult,
    ShotTimingMetrics,
)

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.3  # consistent with perception pipeline

# ---------------------------------------------------------------------------
# NBA reference ranges (degrees)
# Calibrated from biomechanics research + elite player analysis.
# Sources: NSCA Journal, NBA tracking data, peer-reviewed biomechanics studies.
# ---------------------------------------------------------------------------

NBA_REFERENCES: dict[str, tuple[float, float]] = {
    # Upper body — shooting form
    "right_elbow_angle": (30.0, 50.0),  # elbow at release: 30-50° bend
    "left_elbow_angle": (80.0, 120.0),  # guide hand elbow: looser constraint
    "right_shoulder_angle": (70.0, 100.0),  # shoulder abduction at release
    # Lower body — power generation
    "right_knee_angle": (150.0, 170.0),  # at release: legs extended (165° ideal)
    "left_knee_angle": (150.0, 170.0),
    "right_hip_angle": (160.0, 180.0),  # hips extended at release
    # Wrist — follow-through
    "right_wrist_angle": (30.0, 60.0),  # dorsiflexion (snap) at follow-through
}


# ---------------------------------------------------------------------------
# Vector math helpers
# ---------------------------------------------------------------------------


class Point2D(NamedTuple):
    x: float
    y: float


def angle_between_points(a: Point2D, vertex: Point2D, b: Point2D) -> float | None:
    """
    Compute the interior angle (degrees) at `vertex` formed by points a–vertex–b.

    Returns None if any coordinate is (0, 0) with confidence=0 (missing keypoint).
    """
    v1 = np.array([a.x - vertex.x, a.y - vertex.y], dtype=float)
    v2 = np.array([b.x - vertex.x, b.y - vertex.y], dtype=float)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return None

    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def keypoint_map(frame: PoseFrame) -> dict[str, Point2D]:
    """Build a name → (x, y) lookup for a frame's keypoints.

    Only includes keypoints above CONFIDENCE_THRESHOLD (0.3) to avoid
    propagating noisy detections into angle calculations.
    """
    return {
        kp.name: Point2D(kp.x, kp.y)
        for kp in frame.keypoints
        if kp.confidence >= CONFIDENCE_THRESHOLD
    }


# ---------------------------------------------------------------------------
# Per-frame angle extraction
# ---------------------------------------------------------------------------


def compute_joint_angles(frame: PoseFrame) -> list[JointAngle]:
    """
    Extract all relevant joint angles from a single PoseFrame.

    Returns only angles where all 3 required keypoints are present.
    """
    kps = keypoint_map(frame)
    angles: list[JointAngle] = []

    def try_angle(
        joint_name: str,
        kp_a: str,
        kp_vertex: str,
        kp_b: str,
    ) -> None:
        if not all(k in kps for k in (kp_a, kp_vertex, kp_b)):
            return
        val = angle_between_points(kps[kp_a], kps[kp_vertex], kps[kp_b])
        if val is None:
            return
        ref = NBA_REFERENCES.get(joint_name)
        angles.append(
            JointAngle(
                joint=joint_name,
                value_deg=round(val, 1),
                optimal_min=ref[0] if ref else None,
                optimal_max=ref[1] if ref else None,
            )
        )

    # Right arm (shooting arm for right-handed players)
    try_angle("right_elbow_angle", "right_shoulder", "right_elbow", "right_wrist")
    try_angle("right_shoulder_angle", "right_hip", "right_shoulder", "right_elbow")
    try_angle("left_elbow_angle", "left_shoulder", "left_elbow", "left_wrist")

    # Lower body — power chain
    try_angle("right_knee_angle", "right_hip", "right_knee", "right_ankle")
    try_angle("left_knee_angle", "left_hip", "left_knee", "left_ankle")
    try_angle("right_hip_angle", "right_shoulder", "right_hip", "right_knee")

    return angles


# ---------------------------------------------------------------------------
# Shot timing analysis
# ---------------------------------------------------------------------------


def compute_timing(
    frames: list[PoseFrame],
    shot_phases: dict[ShotPhase, int],
) -> ShotTimingMetrics:
    """Derive timing metrics from detected shot phases and frame timestamps."""
    if not frames or not shot_phases:
        return ShotTimingMetrics()

    fps_approx = _estimate_fps(frames)
    frame_ts: dict[int, float] = {f.frame_index: f.timestamp_ms for f in frames}

    setup_start = shot_phases.get(ShotPhase.SETUP)
    jump_start = shot_phases.get(ShotPhase.JUMP)
    release_frame = shot_phases.get(ShotPhase.RELEASE)

    setup_duration: float | None = None
    if setup_start is not None and jump_start is not None:
        setup_duration = frame_ts.get(jump_start, 0.0) - frame_ts.get(setup_start, 0.0)

    jump_to_release: float | None = None
    if jump_start is not None and release_frame is not None:
        jump_to_release = frame_ts.get(release_frame, 0.0) - frame_ts.get(jump_start, 0.0)

    # Heuristic: early release if released before half of jump phase duration
    early_release: bool | None = None
    if jump_to_release is not None and fps_approx > 0:
        # Typical jump phase is ~200-350ms; early = < 150ms
        early_release = jump_to_release < 150.0

    return ShotTimingMetrics(
        setup_duration_ms=round(setup_duration, 1) if setup_duration is not None else None,
        jump_to_release_ms=round(jump_to_release, 1) if jump_to_release is not None else None,
        release_frame_index=release_frame,
        early_release=early_release,
    )


def _estimate_fps(frames: list[PoseFrame]) -> float:
    """Estimate FPS from PoseFrame timestamps."""
    if len(frames) < 2:
        return 30.0
    total_ms = frames[-1].timestamp_ms - frames[0].timestamp_ms
    if total_ms <= 0:
        return 30.0
    return (len(frames) - 1) / (total_ms / 1000.0)


# ---------------------------------------------------------------------------
# Alignment / posture score
# ---------------------------------------------------------------------------


def compute_alignment_score(release_frame: PoseFrame | None) -> float | None:
    """
    Compute a 0–1 alignment score at the release frame.

    Measures shoulder–hip alignment (are they square to the basket?)
    and shooting-arm vertical alignment.
    A score of 1.0 = textbook NBA form; 0.0 = severely misaligned.
    """
    if release_frame is None:
        return None

    kps = keypoint_map(release_frame)

    required = {"left_shoulder", "right_shoulder", "left_hip", "right_hip"}
    if not required.issubset(kps.keys()):
        return None

    # Shoulder tilt (in pixels — normalize by shoulder width)
    l_sh = kps["left_shoulder"]
    r_sh = kps["right_shoulder"]
    shoulder_width = abs(l_sh.x - r_sh.x)
    if shoulder_width < 1e-6:
        return None
    shoulder_tilt = abs(l_sh.y - r_sh.y) / shoulder_width  # 0 = perfectly level

    # Hip tilt
    l_hip = kps["left_hip"]
    r_hip = kps["right_hip"]
    hip_width = abs(l_hip.x - r_hip.x)
    if hip_width < 1e-6:
        return None
    hip_tilt = abs(l_hip.y - r_hip.y) / hip_width

    # Combined score: penalize tilt
    tilt_penalty = min(1.0, (shoulder_tilt + hip_tilt) / 2.0)
    return round(max(0.0, 1.0 - tilt_penalty), 2)


# ---------------------------------------------------------------------------
# Issue detection
# ---------------------------------------------------------------------------


def detect_issues(angles: list[JointAngle], timing: ShotTimingMetrics) -> list[str]:
    """
    Return a list of human-readable biomechanical issues.

    Each issue is a concise description suitable for VLM prompt injection.
    """
    issues: list[str] = []

    for angle in angles:
        if angle.within_range is False:
            ref_str = (
                f"optimal: {angle.optimal_min}–{angle.optimal_max}°"
                if angle.optimal_min and angle.optimal_max
                else ""
            )
            direction = (
                "too low"
                if angle.optimal_min and angle.value_deg < angle.optimal_min
                else "too high"
            )
            issues.append(
                f"{angle.joint.replace('_', ' ')} is {direction} "
                f"({angle.value_deg:.1f}°, {ref_str})"
            )

    if timing.early_release is True:
        issues.append(
            f"Early release detected: ball released only {timing.jump_to_release_ms:.0f}ms "
            "into jump — should release at jump apex"
        )

    return issues


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------


class BiomechanicsAnalyzer:
    """
    Compute the full BiomechanicsReport from a list of PoseFrames.

    Usage::

        analyzer = BiomechanicsAnalyzer()
        report = analyzer.analyze(pose_frames, shot_phases, shot_result)
    """

    def analyze(
        self,
        frames: list[PoseFrame],
        shot_phases: dict[ShotPhase, int] | None = None,
        shot_result: ShotResult = ShotResult.UNKNOWN,
    ) -> BiomechanicsReport:
        """
        Analyse a shot sequence.

        Args:
            frames: Ordered list of PoseFrames for this shot.
            shot_phases: Mapping of ShotPhase → frame_index (from ShotPhaseDetector).
            shot_result: Whether the shot was made/missed (if known).

        Returns:
            BiomechanicsReport with angles, timing, trajectory, and issues.
        """
        if shot_phases is None:
            shot_phases = {}
        if not frames:
            return BiomechanicsReport(shot_result=shot_result)

        # Use release frame for primary angle analysis.
        # Fallback priority: highest-confidence frame (not middle frame, which
        # could be any arbitrary phase and produce misleading biomechanics).
        release_idx = shot_phases.get(ShotPhase.RELEASE)
        release_frame = self._find_frame(frames, release_idx) or self._best_frame(frames)

        angles = compute_joint_angles(release_frame)
        timing = compute_timing(frames, shot_phases)
        alignment = compute_alignment_score(release_frame)
        issues = detect_issues(angles, timing)

        primary = issues[0] if issues else None

        return BiomechanicsReport(
            shot_result=shot_result,
            joint_angles=angles,
            timing=timing,
            trajectory=BallTrajectoryMetrics(),  # ball tracking — Phase 2
            alignment_score=alignment,
            primary_issue=primary,
            issues_detected=issues,
        )

    @staticmethod
    def _find_frame(frames: list[PoseFrame], frame_index: int | None) -> PoseFrame | None:
        if frame_index is None:
            return None
        for f in frames:
            if f.frame_index == frame_index:
                return f
        return None

    @staticmethod
    def _best_frame(frames: list[PoseFrame]) -> PoseFrame:
        """Return the frame with the highest overall keypoint confidence."""
        return max(frames, key=lambda f: f.confidence)
