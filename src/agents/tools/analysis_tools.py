"""Tool function: compute biomechanical metrics from pose data.

Wraps the AI Shoot Phase 1 BiomechanicsAnalyzer.
Uses deferred imports so the module loads in CI without heavy dependencies.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_biomechanics(perception_output_json: str) -> dict[str, Any]:
    """
    Compute biomechanical metrics from pose keypoints.

    Analyzes joint angles, shot timing, ball trajectory, and body alignment
    to produce a structured BiomechanicsReport.

    Args:
        perception_output_json: JSON string matching PerceptionOutput schema
            (output from extract_shot_frames).

    Returns:
        JSON-serializable dict matching BiomechanicsReport schema:
        {shot_result, joint_angles, timing, trajectory, alignment_score,
         primary_issue, issues_detected}. On error, includes an "error" key.
    """
    try:
        perception_data = json.loads(perception_output_json)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON input: {e}"}

    try:
        from src.analysis.biomechanics import BiomechanicsAnalyzer
        from src.api.schemas.domain import PerceptionOutput

        perception = PerceptionOutput.model_validate(perception_data)
        analyzer = BiomechanicsAnalyzer()
        report = analyzer.analyze(  # type: ignore[call-arg]
            perception.key_frames,
            shot_phases=perception.shot_phases or None,
        )
        return report.model_dump()
    except ImportError:
        logger.warning("Analysis modules not available — returning stub biomechanics report")
        return _stub_biomechanics_report()
    except Exception as e:
        logger.error("Biomechanics analysis failed: %s", e)
        return {"error": str(e)}


def _stub_biomechanics_report() -> dict[str, Any]:
    """Stub BiomechanicsReport for testing without heavy model dependencies."""
    return {
        "shot_result": "unknown",
        "joint_angles": [],
        "timing": {},
        "trajectory": {"trajectory_detected": False},
        "alignment_score": None,
        "primary_issue": None,
        "issues_detected": [],
    }
