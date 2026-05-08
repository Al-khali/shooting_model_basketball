#!/usr/bin/env python3
"""
Pose estimation benchmark — Phase 1, Issue #10.

Compares YOLOv11-pose vs MediaPipe BlazePose on basketball videos.
Reports: keypoint accuracy, inference speed (fps), and consistency.

Usage:
    uv run scripts/benchmark_pose.py --video path/to/shot.mp4
    uv run scripts/benchmark_pose.py --video path/to/shot.mp4 --models yolo mediapipe
    uv run scripts/benchmark_pose.py --synthetic  # generate synthetic test frames

Output:
    Markdown report in docs/benchmark_pose_<timestamp>.md
    Console summary table

Tracking: https://github.com/Al-khali/shooting_model_basketball/issues/10
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schemas.domain import PoseModel  # noqa: E402
from src.perception.pose_estimator import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    BasePoseEstimator,
    PoseEstimatorFactory,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------


@dataclass
class ModelBenchmarkResult:
    model: PoseModel
    n_frames: int = 0
    n_frames_with_person: int = 0
    total_inference_ms: float = 0.0
    keypoint_confidence_sums: dict[str, float] = field(default_factory=dict)
    keypoint_detection_counts: dict[str, int] = field(default_factory=dict)
    errors: int = 0

    @property
    def avg_fps(self) -> float:
        if self.total_inference_ms == 0:
            return 0.0
        return self.n_frames / (self.total_inference_ms / 1000.0)

    @property
    def detection_rate(self) -> float:
        if self.n_frames == 0:
            return 0.0
        return self.n_frames_with_person / self.n_frames

    @property
    def avg_inference_ms(self) -> float:
        if self.n_frames == 0:
            return 0.0
        return self.total_inference_ms / self.n_frames

    def avg_confidence(self, keypoint_name: str) -> float:
        count = self.keypoint_detection_counts.get(keypoint_name, 0)
        if count == 0:
            return 0.0
        return self.keypoint_confidence_sums.get(keypoint_name, 0.0) / count


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def benchmark_estimator(
    estimator: BasePoseEstimator,
    frames: list[np.ndarray],
    warmup_frames: int = 5,
) -> ModelBenchmarkResult:
    """Run inference on all frames and collect metrics."""
    result = ModelBenchmarkResult(model=estimator.model_name())

    # Warmup
    logger.info("Warming up %s...", estimator.model_name())
    for frame in frames[:warmup_frames]:
            with contextlib.suppress(Exception):
                estimator.estimate(frame)

    logger.info("Benchmarking %s on %d frames...", estimator.model_name(), len(frames))

    for frame in frames:
        result.n_frames += 1
        t0 = time.perf_counter()
        try:
            keypoints = estimator.estimate(frame)
        except Exception as e:
            logger.warning("Inference error: %s", e)
            result.errors += 1
            result.total_inference_ms += (time.perf_counter() - t0) * 1000
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000
        result.total_inference_ms += elapsed_ms

        if keypoints:
            result.n_frames_with_person += 1
            for kp in keypoints:
                if kp.confidence >= CONFIDENCE_THRESHOLD:
                    result.keypoint_confidence_sums[kp.name] = (
                        result.keypoint_confidence_sums.get(kp.name, 0.0) + kp.confidence
                    )
                    result.keypoint_detection_counts[kp.name] = (
                        result.keypoint_detection_counts.get(kp.name, 0) + 1
                    )

    return result


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------


def load_frames_from_video(
    path: str | Path,
    max_frames: int = 200,
    sample_every: int = 2,
) -> tuple[list[np.ndarray], float]:
    """Load frames from a video file. Returns (frames, fps)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            frames.append(frame)
        idx += 1
        if len(frames) >= max_frames:
            break

    cap.release()
    logger.info("Loaded %d frames from %s (fps=%.1f)", len(frames), path, fps)
    return frames, fps


def generate_synthetic_frames(
    n: int = 100,
    height: int = 480,
    width: int = 640,
) -> list[np.ndarray]:
    """
    Generate synthetic basketball-like frames for CI-safe benchmarking.

    Creates frames with a moving person silhouette (simple rectangle + ellipse)
    to test that the pipeline processes frames without crashing.
    NOTE: No real pose will be detected — confidence scores will be near 0.
    """
    logger.info("Generating %d synthetic frames (%dx%d)", n, width, height)
    frames: list[np.ndarray] = []
    rng = np.random.default_rng(42)

    for i in range(n):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Simulate a court (green floor)
        frame[height // 2:, :] = (34, 100, 34)
        # Simulate player (white body)
        cx = width // 2 + int(20 * np.sin(i * 0.1))
        cy = height // 2 - 30 + int(15 * abs(np.sin(i * 0.2)))  # jumping motion
        cv2.rectangle(frame, (cx - 20, cy - 60), (cx + 20, cy + 30), (220, 220, 220), -1)
        cv2.ellipse(frame, (cx, cy - 70), (18, 18), 0, 0, 360, (200, 160, 120), -1)
        # Add some noise
        noise = rng.integers(0, 20, frame.shape, dtype=np.uint8)
        frame = np.clip(frame.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        frames.append(frame)

    return frames


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


BASKETBALL_KEYPOINTS = [
    "right_shoulder", "left_shoulder",
    "right_elbow", "left_elbow",
    "right_wrist", "left_wrist",
    "right_hip", "left_hip",
    "right_knee", "left_knee",
]


def print_summary(results: list[ModelBenchmarkResult]) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 70)
    print("POSE ESTIMATION BENCHMARK — AI Shoot Phase 1")
    print("=" * 70)

    headers = ["Model", "FPS", "ms/frame", "Detection%", "Errors"]
    rows = [
        [
            r.model.value,
            f"{r.avg_fps:.1f}",
            f"{r.avg_inference_ms:.1f}",
            f"{r.detection_rate * 100:.0f}%",
            str(r.errors),
        ]
        for r in results
    ]

    col_widths = [max(len(h), max((len(row[i]) for row in rows), default=0)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*row))

    print("\nKey basketball keypoints — average confidence (detected frames):")
    kp_headers = ["Keypoint"] + [r.model.value for r in results]
    kp_col_widths = [20] + [15] * len(results)
    kp_fmt = "  ".join(f"{{:<{w}}}" for w in kp_col_widths)
    print(kp_fmt.format(*kp_headers))
    print("  ".join("-" * w for w in kp_col_widths))
    for kp in BASKETBALL_KEYPOINTS:
        row = [kp] + [f"{r.avg_confidence(kp):.2f}" for r in results]
        print(kp_fmt.format(*row))

    print("\n" + "=" * 70)


def save_markdown_report(
    results: list[ModelBenchmarkResult],
    output_dir: Path,
    video_source: str,
) -> Path:
    """Save detailed Markdown report for the benchmark."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"benchmark_pose_{timestamp}.md"
    output_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Pose Estimation Benchmark Report",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Source**: `{video_source}`  ",
        "**Tracking issue**: [#10](https://github.com/Al-khali/shooting_model_basketball/issues/10)",
        "",
        "## Summary",
        "",
        "| Model | FPS | ms/frame | Detection Rate | Errors |",
        "|-------|-----|----------|----------------|--------|",
    ]
    for r in results:
        lines.append(
            f"| {r.model.value} "
            f"| {r.avg_fps:.1f} "
            f"| {r.avg_inference_ms:.1f} "
            f"| {r.detection_rate * 100:.0f}% "
            f"| {r.errors} |"
        )

    lines += [
        "",
        "## Basketball Keypoints — Average Confidence",
        "",
        "| Keypoint | " + " | ".join(r.model.value for r in results) + " |",
        "|----------|" + "|".join(["----"] * len(results)) + "|",
    ]
    for kp in BASKETBALL_KEYPOINTS:
        row = " | ".join(f"{r.avg_confidence(kp):.2f}" for r in results)
        lines.append(f"| {kp} | {row} |")

    lines += ["", "## Recommendation", "", "<!-- Fill in after reviewing the results -->", ""]

    out_path.write_text("\n".join(lines))
    logger.info("Report saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark pose estimators for AI Shoot — Phase 1"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Path to basketball video file")
    source.add_argument("--synthetic", action="store_true", help="Use synthetic test frames")

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["yolo", "mediapipe"],
        default=["yolo"],
        help="Models to benchmark (default: yolo)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=200, help="Max frames to process per model"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs"),
        help="Directory for the Markdown report",
    )
    parser.add_argument("--device", default="cpu", help="Inference device (cpu/cuda/mps)")
    return parser.parse_args()


MODEL_ALIASES: dict[str, PoseModel] = {
    "yolo": PoseModel.YOLOV11,
    "mediapipe": PoseModel.MEDIAPIPE,
}


def main() -> None:
    args = parse_args()

    # Load frames
    if args.synthetic:
        frames = generate_synthetic_frames(n=args.max_frames)
        video_source = "synthetic"
    else:
        frames, _ = load_frames_from_video(args.video, max_frames=args.max_frames)
        video_source = str(args.video)

    if not frames:
        logger.error("No frames loaded — exiting")
        sys.exit(1)

    # Run benchmark
    results: list[ModelBenchmarkResult] = []
    for model_alias in args.models:
        model = MODEL_ALIASES[model_alias]
        try:
            estimator = PoseEstimatorFactory.create(model, device=args.device)
            result = benchmark_estimator(estimator, frames)
            results.append(result)
        except ImportError as e:
            logger.error("Cannot load %s: %s", model.value, e)
        except Exception:
            logger.exception("Benchmark failed for %s", model.value)

    if not results:
        logger.error("No models benchmarked successfully")
        sys.exit(1)

    # Output
    print_summary(results)
    report_path = save_markdown_report(results, args.output_dir, video_source)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
