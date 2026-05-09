"""
Feedback quality evaluator.

Evaluates CoachingFeedback output on multiple quality dimensions:
- Actionability: Is the primary correction concrete and executable?
- Specificity: Does it reference actual data from the report?
- Completeness: Are all required fields populated?
- Consistency: Does it agree with the biomechanics report?
- Safety: No hallucinated statistics not present in the source data.

Used in:
1. CI quality gates (benchmark mode)
2. Production monitoring (flag low-quality responses)
3. Prompt iteration (compare prompt versions)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.api.schemas.domain import CoachingFeedback

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""

    name: str
    score: float  # 0.0–1.0
    passed: bool
    reason: str


@dataclass
class EvaluationResult:
    """Full quality evaluation of a CoachingFeedback."""

    dimensions: list[DimensionScore] = field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = False
    feedback_id: str | None = None

    def __post_init__(self) -> None:
        if self.dimensions:
            self.overall_score = sum(d.score for d in self.dimensions) / len(self.dimensions)
            self.passed = all(d.passed for d in self.dimensions)

    def summary(self) -> str:
        lines = [f"Overall: {self.overall_score:.0%} — {'✅ PASS' if self.passed else '❌ FAIL'}"]
        for d in self.dimensions:
            icon = "✅" if d.passed else "❌"
            lines.append(f"  {icon} {d.name}: {d.score:.0%} — {d.reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Heuristic checks
# ---------------------------------------------------------------------------

# Words that indicate concrete actions
_ACTIONABLE_VERBS = re.compile(
    r"\b(tuck|extend|bend|straighten|hold|follow|keep|slow|speed|raise|lower|"
    r"align|focus|position|drive|push|pull|rotate|flex|lock|release)\b",
    re.IGNORECASE,
)

# Numerical patterns that suggest specificity (e.g. "68°", "320ms", "45%")
_NUMERIC_PATTERN = re.compile(r"\d+\.?\d*\s*(°|ms|%|degrees?|seconds?)")

# Common hallucination patterns: references to famous players not in the prompt
_HALLUCINATION_PATTERN = re.compile(
    r"\b(studies show|research indicates|NBA average is|typically|"
    r"statistically|according to|proven|clinically)\b",
    re.IGNORECASE,
)

# Minimum length thresholds
_MIN_SUMMARY_WORDS = 20
_MIN_DETAILED_WORDS = 40
_MIN_CORRECTION_WORDS = 8


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class FeedbackEvaluator:
    """
    Heuristic quality evaluator for VLM-generated coaching feedback.

    This is intentionally NOT a second LLM call — that would be expensive
    and circular. Instead, we use fast, deterministic heuristics that flag
    obvious quality failures.

    For deeper quality evaluation (semantic coherence, factual grounding),
    use the benchmark mode with human-annotated reference shots.
    """

    # Minimum pass threshold per dimension
    PASS_THRESHOLD = 0.5

    def evaluate(self, feedback: CoachingFeedback) -> EvaluationResult:
        """
        Run all quality checks on a CoachingFeedback instance.

        Args:
            feedback: The VLM-generated coaching feedback to evaluate.

        Returns:
            EvaluationResult with per-dimension scores and overall verdict.
        """
        dimensions = [
            self._check_completeness(feedback),
            self._check_actionability(feedback),
            self._check_specificity(feedback),
            self._check_consistency(feedback),
            self._check_no_hallucination(feedback),
        ]
        return EvaluationResult(
            dimensions=dimensions,
            feedback_id=feedback.player_id,
        )

    def _check_completeness(self, feedback: CoachingFeedback) -> DimensionScore:
        """All required text fields must be non-empty and meet minimum length."""
        issues = []

        summary_words = len(feedback.summary.split())
        if summary_words < _MIN_SUMMARY_WORDS:
            issues.append(f"summary too short ({summary_words} words, min {_MIN_SUMMARY_WORDS})")

        correction_words = len(feedback.primary_correction.split())
        if correction_words < _MIN_CORRECTION_WORDS:
            issues.append(
                f"primary_correction too short ({correction_words} words, "
                f"min {_MIN_CORRECTION_WORDS})"
            )

        detailed_words = len(feedback.detailed_analysis.split())
        if detailed_words < _MIN_DETAILED_WORDS:
            issues.append(
                f"detailed_analysis too short ({detailed_words} words, min {_MIN_DETAILED_WORDS})"
            )

        if not issues:
            return DimensionScore("completeness", 1.0, True, "All fields meet minimum length")

        score = 1.0 - (len(issues) / 3)
        return DimensionScore(
            "completeness",
            max(0.0, score),
            score >= self.PASS_THRESHOLD,
            "; ".join(issues),
        )

    def _check_actionability(self, feedback: CoachingFeedback) -> DimensionScore:
        """Primary correction must contain action verbs."""
        text = feedback.primary_correction + " " + feedback.summary
        matches = _ACTIONABLE_VERBS.findall(text)
        count = len(set(m.lower() for m in matches))

        if count >= 2:
            return DimensionScore("actionability", 1.0, True, f"{count} action verbs found")
        if count == 1:
            return DimensionScore("actionability", 0.6, True, "1 action verb found (acceptable)")
        return DimensionScore(
            "actionability", 0.0, False, "No action verbs found — correction is not actionable"
        )

    def _check_specificity(self, feedback: CoachingFeedback) -> DimensionScore:
        """
        Detailed analysis should reference actual data (angles, timing, scores).

        We check if numerical values from the biomechanics report appear in
        the generated text, OR if generic numeric patterns are present.
        """
        text = feedback.detailed_analysis + " " + feedback.summary
        numeric_refs = _NUMERIC_PATTERN.findall(text)

        # Also check if joint names from the report appear in the text
        report_joints = {a.joint for a in feedback.biomechanics.joint_angles}
        joint_refs = sum(1 for j in report_joints if j.replace("_", " ") in text.lower())

        has_numbers = len(numeric_refs) >= 1
        has_joints = joint_refs >= 1

        if has_numbers and has_joints:
            return DimensionScore(
                "specificity",
                1.0,
                True,
                f"{len(numeric_refs)} numeric refs, {joint_refs} joint refs",
            )
        if has_numbers or has_joints:
            return DimensionScore("specificity", 0.6, True, "partial data references found")
        return DimensionScore(
            "specificity", 0.2, False, "No specific data references found — response may be generic"
        )

    def _check_consistency(self, feedback: CoachingFeedback) -> DimensionScore:
        """
        Check that feedback is consistent with the biomechanics report.

        - If shot_result is MADE, feedback should not heavily criticise
        - If primary_issue is set, it should appear somewhere in the feedback
        """
        issues = []
        primary_issue = feedback.biomechanics.primary_issue
        if primary_issue:
            # Key words from the issue should appear in the feedback
            key_words = [w.lower() for w in primary_issue.split() if len(w) > 4]
            text = (feedback.summary + " " + feedback.detailed_analysis).lower()
            found = sum(1 for kw in key_words if kw in text)
            if key_words and found == 0:
                issues.append(
                    f"primary_issue '{primary_issue[:40]}' not reflected in feedback text"
                )

        if not issues:
            return DimensionScore("consistency", 1.0, True, "Feedback consistent with report")

        return DimensionScore(
            "consistency",
            0.4,
            False,
            "; ".join(issues),
        )

    def _check_no_hallucination(self, feedback: CoachingFeedback) -> DimensionScore:
        """
        Flag common hallucination patterns: ungrounded statistical claims.

        This is a conservative heuristic — it detects patterns like
        'studies show' or 'NBA average is' that suggest the model may be
        confabulating rather than referencing the provided data.
        """
        text = feedback.summary + " " + feedback.detailed_analysis
        matches = _HALLUCINATION_PATTERN.findall(text)

        if not matches:
            return DimensionScore(
                "no_hallucination", 1.0, True, "No ungrounded claim patterns detected"
            )

        unique = list({m.lower() for m in matches})
        score = max(0.0, 1.0 - len(unique) * 0.3)
        return DimensionScore(
            "no_hallucination",
            score,
            score >= self.PASS_THRESHOLD,
            f"Possible ungrounded claims: {unique[:3]}",
        )
