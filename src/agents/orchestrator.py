"""
AI Shoot Agent Pipeline — Orchestrator.

Provides two execution modes:

1. ``ShotAnalysisPipeline`` — framework-independent (no ADK required).
   Suitable for POC, local dev, testing, and simple Cloud Run deployments.

2. ``create_adk_pipeline()`` — builds a Google ADK SequentialAgent.
   Requires ``google-adk`` to be installed. Falls back gracefully to
   ``ShotAnalysisPipeline`` if ADK is not available.

The two interfaces are API-compatible: both accept a ``VideoInput`` and
return the same result structure so callers need not change when upgrading.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.agents.memory import PlayerMemoryService
from src.agents.tools.analysis_tools import compute_biomechanics
from src.agents.tools.coach_tools import generate_coaching_feedback
from src.agents.tools.perception_tools import extract_shot_frames
from src.agents.tools.planner_tools import build_training_plan
from src.api.schemas.domain import CoachingFeedback, VideoInput

logger = logging.getLogger(__name__)


class ShotAnalysisPipeline:
    """
    Orchestrates the full AI Shoot analysis pipeline without Google ADK.

    Runs the perception → biomechanics → VLM coaching → training plan chain,
    using ``PlayerMemoryService`` for per-player session context and history.

    In production this maps to an ADK ``SequentialAgent`` with Agent Runtime's
    Session Service and Memory Bank. The tool functions used here are the same
    ones wrapped with ``@adk_tool`` in ``create_adk_pipeline()``.

    Usage::

        pipeline = ShotAnalysisPipeline()
        result = pipeline.analyze(VideoInput(video_path="shot.mp4", player_id="p42"))
    """

    def __init__(self, memory_service: PlayerMemoryService | None = None) -> None:
        self.memory = memory_service or PlayerMemoryService()

    def analyze(self, video_input: VideoInput) -> dict[str, Any]:
        """
        Run the full analysis pipeline synchronously.

        Returns a dict with keys:
        - ``player_id``
        - ``coaching_feedback`` — CoachingFeedback dict (or None on error)
        - ``training_plan`` — training plan dict (or None on error)
        - ``processing_time_ms``
        - ``session_number``
        - ``error`` — present only on fatal failure
        """
        start_ms = time.monotonic() * 1000
        player_id = video_input.player_id or "anonymous"

        # Step 1 — Load player context (mirrors ADK Session Service)
        context = self.memory.build_context(
            player_id=player_id,
            player_level=video_input.player_level,
            notes=video_input.notes,
        )
        logger.info(
            "Pipeline start — player=%s level=%s session=#%d",
            player_id,
            context.player_level,
            context.session_count,
        )

        # Step 2 — Perception
        perception_result = extract_shot_frames(video_input.video_path)
        if "error" in perception_result:
            logger.warning("Perception error: %s", perception_result["error"])

        # Step 3 — Biomechanics
        biomechanics_result = compute_biomechanics(json.dumps(perception_result))
        if "error" in biomechanics_result:
            logger.warning("Biomechanics error: %s", biomechanics_result["error"])

        # Step 4 — VLM coaching feedback
        feedback_result = generate_coaching_feedback(
            biomechanics_json=json.dumps(biomechanics_result),
            player_id=player_id,
            player_level=context.player_level.value,
            session_count=context.session_count,
            recurring_issues_json=json.dumps(context.recurring_issues),
            previous_drills_json=json.dumps(context.previous_drills),
        )
        if "error" in feedback_result and not feedback_result.get("summary"):
            logger.error("Coaching fatal error: %s", feedback_result["error"])
            elapsed_ms = time.monotonic() * 1000 - start_ms
            return self._error_result(player_id, feedback_result["error"], elapsed_ms)

        # Step 5 — Training plan
        plan = build_training_plan(
            feedback_json=json.dumps(feedback_result),
            session_count=context.session_count,
            recurring_issues_json=json.dumps(context.recurring_issues),
            previous_drills_json=json.dumps(context.previous_drills),
        )

        # Step 6 — Persist session to memory (only for identified players)
        if video_input.player_id:
            try:
                feedback = CoachingFeedback.model_validate(feedback_result)
                self.memory.record_feedback(player_id, feedback, video_input.player_level)
                logger.info(
                    "Session #%d recorded for player %s", context.session_count + 1, player_id
                )
            except Exception as e:
                logger.warning("Could not persist session for %s: %s", player_id, e)

        elapsed_ms = time.monotonic() * 1000 - start_ms
        logger.info("Pipeline complete in %.0f ms", elapsed_ms)

        return {
            "player_id": player_id,
            "coaching_feedback": feedback_result,
            "training_plan": plan,
            "processing_time_ms": elapsed_ms,
            "session_number": context.session_count + 1,
        }

    @staticmethod
    def _error_result(player_id: str, error: str, elapsed_ms: float) -> dict[str, Any]:
        return {
            "player_id": player_id,
            "error": error,
            "coaching_feedback": None,
            "training_plan": None,
            "processing_time_ms": elapsed_ms,
        }


def create_adk_pipeline() -> Any:
    """
    Build the production Google ADK ``SequentialAgent`` pipeline.

    Wraps the same tool functions used by ``ShotAnalysisPipeline`` with the
    ADK ``@tool`` decorator and composes them into four specialised LlmAgents:

    - ``PerceiverAgent`` — Gemini Flash Lite, runs frame extraction
    - ``AnalyzerAgent`` — Gemini Flash, interprets biomechanics
    - ``CoachAgent`` — Gemini Flash, generates coaching feedback
    - ``PlannerAgent`` — Gemini Flash, builds the training plan

    Falls back to ``ShotAnalysisPipeline`` if ``google-adk`` is not installed.

    Returns:
        ``google.adk.agents.SequentialAgent`` or ``ShotAnalysisPipeline``.
    """
    try:
        from google.adk.agents import LlmAgent, SequentialAgent  # type: ignore[import]
        from google.adk.tools import tool as adk_tool  # type: ignore[import]
    except ImportError:
        logger.warning(
            "google-adk not installed — falling back to ShotAnalysisPipeline. "
            "Install with: pip install google-adk"
        )
        return ShotAnalysisPipeline()

    adk_extract = adk_tool(extract_shot_frames)
    adk_biomechanics = adk_tool(compute_biomechanics)
    adk_coaching = adk_tool(generate_coaching_feedback)
    adk_plan = adk_tool(build_training_plan)

    perceiver_agent = LlmAgent(
        name="PerceiverAgent",
        model="gemini-2.5-flash-lite-001",
        description="Extracts basketball shot frames and estimates player pose.",
        instruction=(
            "You are a computer vision agent for basketball analysis. "
            "When given a video path, call extract_shot_frames to extract pose data. "
            "Output the result as JSON for the next agent."
        ),
        tools=[adk_extract],
    )

    analyzer_agent = LlmAgent(
        name="AnalyzerAgent",
        model="gemini-2.5-flash-001",
        description="Analyzes biomechanical metrics from pose data.",
        instruction=(
            "You are a biomechanics expert agent. "
            "Given pose data JSON from PerceiverAgent, call compute_biomechanics. "
            "Summarize the key findings for the CoachAgent."
        ),
        tools=[adk_biomechanics],
    )

    coach_agent = LlmAgent(
        name="CoachAgent",
        model="gemini-2.5-flash-001",
        description="Generates personalized basketball coaching feedback.",
        instruction=(
            "You are an expert basketball coach agent. "
            "Given biomechanics data and player context from session state, "
            "call generate_coaching_feedback to produce actionable coaching. "
            "Prioritize recurring issues when present."
        ),
        tools=[adk_coaching],
    )

    planner_agent = LlmAgent(
        name="PlannerAgent",
        model="gemini-2.5-flash-001",
        description="Builds personalized weekly training plans.",
        instruction=(
            "You are a basketball training planner agent. "
            "Given coaching feedback, build a weekly training plan that progresses "
            "the player, avoids drill repetition, and addresses recurring issues first. "
            "Call build_training_plan with the feedback data."
        ),
        tools=[adk_plan],
    )

    return SequentialAgent(
        name="ShotAnalysisPipeline",
        description=(
            "End-to-end basketball shot analysis: "
            "video → perception → biomechanics → coaching → training plan."
        ),
        sub_agents=[perceiver_agent, analyzer_agent, coach_agent, planner_agent],
    )
