"""
Basketball VLM Analyzer.

Orchestrates the full pipeline:
  BiomechanicsReport + SessionContext → VLM prompt → CoachingFeedback

This is the main entry point for Phase 2 intelligence.
"""

from __future__ import annotations

import logging
import time

from src.api.schemas.domain import (
    BiomechanicsReport,
    CoachingFeedback,
    Drill,
    PlayerLevel,
    SessionContext,
    ShotResult,
)
from src.vlm.base import BaseVLMClient, Message, VLMError, VLMParseError
from src.vlm.prompts.basketball import (
    BASKETBALL_COACH_SYSTEM_PROMPT,
    PROMPT_VERSION,
    build_analysis_prompt,
)

logger = logging.getLogger(__name__)

# Minimum acceptable confidence from VLM (below this → mark as low quality)
MIN_VLM_CONFIDENCE = 0.3


class BasketballVLMAnalyzer:
    """
    Converts a BiomechanicsReport into actionable CoachingFeedback via VLM.

    The analyzer is intentionally thin — it does not do biomechanics computation
    (that's BiomechanicsAnalyzer's job). It purely translates structured data
    into human-readable, actionable coaching language.

    Usage::

        from src.vlm.gemini_client import GeminiFlashClient
        client = GeminiFlashClient.from_env()
        analyzer = BasketballVLMAnalyzer(client)
        feedback = analyzer.analyze(report, context)
    """

    def __init__(self, client: BaseVLMClient) -> None:
        self.client = client

    def analyze(
        self,
        report: BiomechanicsReport,
        context: SessionContext | None = None,
        shot_number: int = 1,
    ) -> CoachingFeedback:
        """
        Generate coaching feedback from a biomechanics report.

        Args:
            report: Structured biomechanics output from BiomechanicsAnalyzer.
            context: Optional player session context. Used for adaptive coaching.
            shot_number: Index of this shot in the session.

        Returns:
            CoachingFeedback with VLM-generated text and structured drills.

        Raises:
            VLMError: If the API call fails.
            VLMParseError: If the response cannot be parsed into CoachingFeedback.
        """
        start_ms = time.monotonic() * 1000

        user_prompt = build_analysis_prompt(report, context, shot_number)
        messages = [
            Message(role="system", content=BASKETBALL_COACH_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]

        logger.info(
            "Calling VLM %s for shot #%d (player=%s)",
            self.client.model_id,
            shot_number,
            context.player_id if context else "anonymous",
        )

        try:
            raw = self.client.complete_json(messages)
        except VLMParseError:
            # JSON mode failed — fallback to text completion + regex extraction
            logger.warning("JSON mode failed, falling back to text completion")
            raw = self._fallback_parse(messages)
        except VLMError as exc:
            # Full API failure — return safe error feedback, don't crash
            logger.error("VLM API error: %s", exc)
            elapsed_ms = time.monotonic() * 1000 - start_ms
            return self._error_feedback(report, context, str(exc))

        elapsed_ms = time.monotonic() * 1000 - start_ms
        logger.info("VLM response received in %.0fms", elapsed_ms)

        return self._build_feedback(raw, report, context, elapsed_ms)

    def _fallback_parse(self, messages: list[Message]) -> dict:
        """
        Fallback: call text completion and extract JSON from the response.
        Used when JSON mode is unavailable or fails.
        """
        import json
        import re

        text = self.client.complete(messages)
        # Extract JSON block from markdown code fences or raw JSON
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Try to find first { ... } block
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if not match:
                raise VLMParseError(f"No JSON found in VLM response: {text[:300]}")
            json_str = match.group(1)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise VLMParseError(f"Could not parse extracted JSON: {exc}") from exc

    def _build_feedback(
        self,
        raw: dict,
        report: BiomechanicsReport,
        context: SessionContext | None,
        elapsed_ms: float,
    ) -> CoachingFeedback:
        """Build CoachingFeedback from raw VLM dict output."""
        # Extract fields with safe fallbacks.
        # Use `or` instead of dict default to handle explicit null values from VLM.
        summary = str(raw.get("summary") or "Analysis not available.")
        primary_correction = str(raw.get("primary_correction") or "No correction identified.")
        detailed_analysis = str(raw.get("detailed_analysis") or summary)

        # Defensive confidence parsing: VLM may return null, a string, or out-of-range float.
        try:
            val = raw.get("confidence")
            confidence = float(val) if val is not None else MIN_VLM_CONFIDENCE
        except (ValueError, TypeError):
            confidence = MIN_VLM_CONFIDENCE
        confidence = max(0.0, min(1.0, confidence))

        # Parse drills — use `or []` to handle explicit null from VLM
        drills: list[Drill] = []
        for d in (raw.get("drills") or [])[:3]:  # max 3 drills
            try:
                level_str = d.get("difficulty", "intermediate")
                try:
                    difficulty = PlayerLevel(level_str)
                except ValueError:
                    difficulty = PlayerLevel.INTERMEDIATE

                # Safely convert duration: VLM may return null, float, or non-digit string
                raw_duration = d.get("duration_minutes")
                try:
                    duration_minutes = int(float(raw_duration)) if raw_duration is not None else 10
                except (ValueError, TypeError):
                    duration_minutes = 10

                drills.append(
                    Drill(
                        name=str(d.get("name") or "Unnamed drill"),
                        description=str(d.get("description") or ""),
                        duration_minutes=duration_minutes,
                        focus=str(d.get("focus") or ""),
                        difficulty=difficulty,
                    )
                )
            except Exception as exc:
                logger.warning("Could not parse drill: %s — %s", d, exc)

        return CoachingFeedback(
            player_id=context.player_id if context else None,
            shot_result=report.shot_result,
            confidence=confidence,
            summary=summary,
            primary_correction=primary_correction,
            detailed_analysis=detailed_analysis,
            biomechanics=report,
            drills=drills,
            model_used=f"{self.client.model_id}|prompts@{PROMPT_VERSION}",
            processing_time_ms=elapsed_ms,
        )

    def analyze_session(
        self,
        reports: list[BiomechanicsReport],
        context: SessionContext | None = None,
    ) -> list[CoachingFeedback]:
        """
        Analyse multiple shots from a session.

        Useful for identifying patterns across shots (e.g. fatigue-induced
        elbow flare appearing in shots 8-10 but not 1-7).

        Args:
            reports: List of BiomechanicsReport, one per shot.
            context: Player session context.

        Returns:
            List of CoachingFeedback, one per shot.
        """
        results = []
        for i, report in enumerate(reports, start=1):
            try:
                feedback = self.analyze(report, context, shot_number=i)
                results.append(feedback)
            except (VLMError, VLMParseError) as exc:
                logger.error("VLM failed for shot #%d: %s", i, exc)
                # Return a minimal feedback rather than crashing the session
                results.append(self._error_feedback(report, context, str(exc)))
        return results

    def _error_feedback(
        self,
        report: BiomechanicsReport,
        context: SessionContext | None,
        error_msg: str,
    ) -> CoachingFeedback:
        """Produce a safe fallback CoachingFeedback when VLM fails."""
        logger.warning("Using error fallback feedback: %s", error_msg)
        primary = report.primary_issue or "Unable to determine primary issue (VLM error)."
        return CoachingFeedback(
            player_id=context.player_id if context else None,
            shot_result=report.shot_result
            if report.shot_result != ShotResult.UNKNOWN
            else ShotResult.UNKNOWN,
            confidence=0.0,
            summary=f"Analysis partially unavailable due to a service error. Primary detected issue: {primary}",
            primary_correction=primary,
            detailed_analysis=f"VLM analysis unavailable: {error_msg}. Structured biomechanics data is still available.",
            biomechanics=report,
            drills=[],
            model_used="fallback",
            processing_time_ms=None,
        )
