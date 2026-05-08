"""
Shot phase detector.

Segments a basketball shot sequence into 4 canonical phases:
  Setup → Jump → Release → Follow-through

Algorithm: heuristic vertical kinematics on hip keypoints.
No ML required — pure geometry on detected poses.
"""

from __future__ import annotations

import logging

import numpy as np

from src.api.schemas.domain import PoseFrame, ShotPhase

logger = logging.getLogger(__name__)

# Keypoints used as reference for vertical position tracking
HIP_KEYPOINTS = ("right_hip", "left_hip")
WRIST_KEYPOINTS = ("right_wrist", "left_wrist")
SHOULDER_KEYPOINTS = ("right_shoulder", "left_shoulder")
ANKLE_KEYPOINTS = ("right_ankle", "left_ankle")

CONFIDENCE_MIN = 0.25  # minimum confidence to use a keypoint

# Smoothing window for vertical position signal
SMOOTH_WINDOW = 3

# Min vertical displacement (px) to classify as "in the air"
MIN_JUMP_DISPLACEMENT_PX = 8.0

# Fraction of jump phase at which release is expected (heuristic)
RELEASE_FRACTION = 0.65  # release happens at ~65% of jump duration (near apex)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_avg_y(frame: PoseFrame, keypoint_names: tuple[str, ...]) -> float | None:
    """Return the mean y-coordinate for a set of keypoints, or None if missing."""
    ys = [
        kp.y
        for kp in frame.keypoints
        if kp.name in keypoint_names and kp.confidence >= CONFIDENCE_MIN
    ]
    return float(np.mean(ys)) if ys else None


def _smooth(values: list[float], window: int = SMOOTH_WINDOW) -> list[float]:
    """Simple moving average smoothing."""
    if len(values) <= window:
        return values
    result: list[float] = []
    half = window // 2
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        result.append(float(np.mean(values[start:end])))
    return result


def _is_wrist_above_head(frame: PoseFrame) -> bool:
    """
    Heuristic: wrist y < shoulder y means wrist is above shoulder in image coords
    (y increases downward in image/screen coordinates).
    """
    wrist_y = _get_avg_y(frame, WRIST_KEYPOINTS)
    shoulder_y = _get_avg_y(frame, SHOULDER_KEYPOINTS)
    if wrist_y is None or shoulder_y is None:
        return False
    return wrist_y < shoulder_y  # in image coords: smaller y = higher up


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


class ShotPhaseDetector:
    """
    Detect shot phases from a sequence of PoseFrames.

    Phase detection algorithm:
    1. Track hip y-position over time
    2. Smooth signal to remove noise
    3. Find the frame where hip starts moving upward significantly → JUMP start
    4. Estimate RELEASE at ~65% of the ascent (near jump apex)
    5. Detect FOLLOW_THROUGH as frames after release where wrist stays high
    6. Everything before JUMP = SETUP

    This is a heuristic approach calibrated for basketball shots;
    for court-level analysis a trained temporal model would be used instead.
    """

    def detect(self, frames: list[PoseFrame]) -> dict[ShotPhase, int]:
        """
        Segment a shot sequence into phases.

        Args:
            frames: Ordered PoseFrames for this shot (already filtered for valid poses).

        Returns:
            Dict mapping ShotPhase → frame_index where that phase begins.
            Only phases that could be detected are included.
        """
        if len(frames) < 4:
            logger.debug("Not enough frames (%d) for phase detection", len(frames))
            return {}

        hip_ys = [_get_avg_y(f, HIP_KEYPOINTS) for f in frames]
        valid_mask = [y is not None for y in hip_ys]

        if sum(valid_mask) < 3:
            logger.debug("Not enough valid hip keypoints for phase detection")
            return {}

        # Fill missing values with linear interpolation
        hip_ys_filled = _fill_missing(hip_ys)  # type: ignore[arg-type]
        hip_ys_smooth = _smooth(hip_ys_filled)

        return self._find_phases(frames, hip_ys_smooth)

    def _find_phases(
        self,
        frames: list[PoseFrame],
        hip_ys: list[float],
    ) -> dict[ShotPhase, int]:
        """Core phase segmentation logic."""
        n = len(frames)
        phases: dict[ShotPhase, int] = {}

        # Compute velocity: positive = moving DOWN (y increases), negative = UP (jumping)
        velocity = [0.0] + [hip_ys[i] - hip_ys[i - 1] for i in range(1, n)]

        # --- SETUP: first frame (always present)
        phases[ShotPhase.SETUP] = frames[0].frame_index

        # --- JUMP: first frame with sustained upward velocity (negative velocity)
        jump_start_local = self._find_jump_start(velocity, hip_ys)
        if jump_start_local is not None:
            phases[ShotPhase.JUMP] = frames[jump_start_local].frame_index

            # --- RELEASE: near apex (velocity crosses from negative to positive)
            release_local = self._find_release(velocity, jump_start_local, frames)
            if release_local is not None:
                phases[ShotPhase.RELEASE] = frames[release_local].frame_index

                # --- FOLLOW_THROUGH: frames after release
                follow_local = min(release_local + 2, n - 1)
                phases[ShotPhase.FOLLOW_THROUGH] = frames[follow_local].frame_index

        # Annotate frames in-place
        for phase, frame_idx in phases.items():
            for f in frames:
                if f.frame_index == frame_idx:
                    f.phase = phase
                    break

        return phases

    @staticmethod
    def _find_jump_start(
        velocity: list[float],
        hip_ys: list[float],
    ) -> int | None:
        """
        Find the first frame index where the jump starts.

        A jump is detected when velocity becomes consistently negative
        (moving upward) for at least 2 consecutive frames.
        """
        n = len(velocity)

        for i in range(1, n - 1):
            if velocity[i] < -1.0 and velocity[i + 1] < 0.0:
                return i
        return None

    @staticmethod
    def _find_release(
        velocity: list[float],
        jump_start: int,
        frames: list[PoseFrame],
    ) -> int | None:
        """
        Find release frame: the apex of the jump where wrist is above head.

        Strategy:
        1. Look for velocity zero-crossing after jump_start (hip apex)
        2. Among those candidates, pick the one where wrist is above shoulder
        """
        n = len(velocity)

        # Find velocity zero crossing (apex)
        apex_candidates: list[int] = []
        for i in range(jump_start + 1, min(n, jump_start + 30)):
            if velocity[i - 1] <= 0.0 and velocity[i] >= 0.0:
                apex_candidates.append(i)

        if not apex_candidates:
            # No clear apex — use RELEASE_FRACTION heuristic
            remaining = n - jump_start
            release_local = jump_start + max(1, int(remaining * RELEASE_FRACTION))
            return min(release_local, n - 1)

        # Prefer the apex candidate where wrist is above shoulder
        for apex in apex_candidates:
            if _is_wrist_above_head(frames[apex]):
                return apex

        return apex_candidates[0]


# ---------------------------------------------------------------------------
# Missing value interpolation
# ---------------------------------------------------------------------------


def _fill_missing(values: list[float | None]) -> list[float]:
    """
    Linear interpolation to fill None gaps in a 1D signal.
    Edges are filled with nearest valid value.
    """
    result = list(values)
    n = len(result)

    # Forward fill start
    first_valid = next((i for i, v in enumerate(result) if v is not None), None)
    if first_valid is None:
        return [0.0] * n
    for i in range(first_valid):
        result[i] = result[first_valid]

    # Backward fill end
    last_valid = n - 1 - next(
        (i for i, v in enumerate(reversed(result)) if v is not None), 0
    )
    for i in range(last_valid + 1, n):
        result[i] = result[last_valid]

    # Interpolate gaps
    i = 0
    while i < n:
        if result[i] is None:
            left = i - 1
            right = i
            while right < n and result[right] is None:
                right += 1
            if right < n:
                left_val = float(result[left])  # type: ignore[arg-type]
                right_val = float(result[right])  # type: ignore[arg-type]
                for j in range(left + 1, right):
                    frac = (j - left) / (right - left)
                    result[j] = left_val + frac * (right_val - left_val)
            i = right
        else:
            i += 1

    return [float(v) for v in result]  # type: ignore[misc]
