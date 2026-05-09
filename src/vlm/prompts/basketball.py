"""
Basketball-specific VLM prompt templates.

Design principles:
- System prompt = elite coach persona with biomechanics background
- User prompt = rich structured context → coaching request
- Output format = JSON matching CoachingFeedback schema
- Prompts are versioned so we can A/B test and track quality over time
"""

from __future__ import annotations

from src.api.schemas.domain import (
    BiomechanicsReport,
    JointAngle,
    PlayerLevel,
    SessionContext,
    ShotResult,
)

PROMPT_VERSION = "v1.0"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

BASKETBALL_COACH_SYSTEM_PROMPT = """\
You are an elite basketball shooting coach with a PhD in sports biomechanics.
You have worked with NBA players (Stephen Curry, Klay Thompson level) and \
collegiate programs.

Your role is to analyse a player's shooting mechanics from biomechanical data \
and provide coaching feedback that is:
1. SPECIFIC — reference exact joint angles, timing values, and phase names
2. ACTIONABLE — give one clear correction the player can apply in their next session
3. PROGRESSIVE — adapt complexity to the player's level (beginner/intermediate/advanced/elite)
4. ENCOURAGING — maintain motivation while being technically precise

You respond ONLY with a JSON object matching the schema provided in each request.
Never add commentary outside the JSON. Never fabricate statistics not present in the data.
If data is insufficient, acknowledge it honestly in the summary field.
"""

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _level_instruction(level: PlayerLevel) -> str:
    """Return coaching style instruction adapted to player level."""
    return {
        PlayerLevel.BEGINNER: (
            "Use simple language. Focus on ONE fundamental fix. "
            "Avoid biomechanics jargon — explain everything in plain terms."
        ),
        PlayerLevel.INTERMEDIATE: (
            "Balance technical precision with accessible language. "
            "Reference 2-3 mechanics but lead with the most impactful change."
        ),
        PlayerLevel.ADVANCED: (
            "Be technically precise. Reference joint angles, phase timing, and kinetic chain. "
            "The player can handle detailed biomechanics vocabulary."
        ),
        PlayerLevel.ELITE: (
            "Full technical depth. Reference NBA standards and research. "
            "Focus on marginal gains — the player has strong fundamentals. "
            "Assume knowledge of kinetic chain, ground reaction force, and shot arc physics."
        ),
    }.get(level, "Balance technical precision with accessible language.")


def _format_angles(angles: list[JointAngle]) -> str:
    """Format joint angles for injection into the prompt."""
    if not angles:
        return "  No joint angle data available."
    lines = []
    for a in angles:
        status = ""
        if a.within_range is True:
            status = "✓ within optimal range"
        elif a.within_range is False:
            ref = f"[optimal: {a.optimal_min}°–{a.optimal_max}°]"
            status = f"⚠ OUTSIDE optimal range {ref}"
        else:
            status = "no reference range"
        lines.append(f"  • {a.joint}: {a.value_deg:.1f}° — {status}")
    return "\n".join(lines)


def _format_issues(report: BiomechanicsReport) -> str:
    """Format detected issues for the prompt."""
    if not report.issues_detected:
        return "  No major issues detected."
    return "\n".join(f"  {i + 1}. {issue}" for i, issue in enumerate(report.issues_detected))


def _format_timing(report: BiomechanicsReport) -> str:
    """Format shot timing data."""
    t = report.timing
    parts = []
    if t.setup_duration_ms is not None:
        parts.append(f"setup phase: {t.setup_duration_ms:.0f}ms")
    if t.jump_to_release_ms is not None:
        parts.append(f"jump-to-release: {t.jump_to_release_ms:.0f}ms")
    if t.early_release is not None:
        parts.append("early release: YES ⚠" if t.early_release else "early release: NO ✓")
    return "  " + " | ".join(parts) if parts else "  Timing data unavailable."


def build_analysis_prompt(
    report: BiomechanicsReport,
    context: SessionContext | None = None,
    shot_number: int = 1,
) -> str:
    """
    Build the user-side prompt for a shot analysis request.

    Args:
        report: Biomechanics analysis output from BiomechanicsAnalyzer.
        context: Optional player session context (history, level, recurring issues).
        shot_number: Shot index within the session (for context).

    Returns:
        Formatted prompt string ready to send to the VLM.
    """
    level = context.player_level if context else PlayerLevel.INTERMEDIATE
    level_instr = _level_instruction(level)

    shot_result_str = {
        ShotResult.MADE: "MADE ✓",
        ShotResult.MISSED: "MISSED ✗",
        ShotResult.UNKNOWN: "unknown (no tracking data)",
    }.get(report.shot_result, "unknown")

    alignment_str = (
        f"{report.alignment_score * 100:.0f}%"
        if report.alignment_score is not None
        else "not computed"
    )

    recurring = ""
    if context and context.recurring_issues:
        issues_list = "\n".join(f"  - {iss}" for iss in context.recurring_issues)
        recurring = f"\nKnown recurring issues from previous sessions:\n{issues_list}"

    previous_drills = ""
    if context and context.previous_drills:
        drills_list = ", ".join(context.previous_drills)
        previous_drills = f"\nDrills already assigned (avoid repeating): {drills_list}"

    session_info = ""
    if context:
        session_info = f"""
PLAYER PROFILE
--------------
Player ID   : {context.player_id}
Level       : {level.value}
Session #   : {context.session_count + 1}
{recurring}{previous_drills}
"""

    return f"""Analyse shot #{shot_number} and provide coaching feedback.

{session_info}
SHOT BIOMECHANICS
-----------------
Shot result        : {shot_result_str}
Overall alignment  : {alignment_str}

Joint angles at release:
{_format_angles(report.joint_angles)}

Shot timing:
{_format_timing(report)}

Detected issues (priority order):
{_format_issues(report)}

COACHING STYLE INSTRUCTION
---------------------------
{level_instr}

REQUIRED JSON OUTPUT FORMAT
---------------------------
Return ONLY a JSON object with this exact structure. No other text.

{{
  "summary": "<2-3 sentence plain-language summary of the shot mechanics>",
  "primary_correction": "<The ONE most important thing to fix, in 1 sentence>",
  "detailed_analysis": "<Full technical analysis, 3-6 sentences, referencing specific data>",
  "confidence": <float 0.0-1.0 representing your confidence in this analysis>,
  "drills": [
    {{
      "name": "<drill name>",
      "description": "<how to perform it>",
      "duration_minutes": <int>,
      "focus": "<what biomechanics this targets>",
      "difficulty": "<beginner|intermediate|advanced|elite>"
    }}
  ]
}}

Provide 1-3 drills. Keep them specific to the detected issues. \
Do NOT recommend generic drills not tied to the data.
"""


# ---------------------------------------------------------------------------
# Few-shot examples (used for quality calibration / evaluation baseline)
# ---------------------------------------------------------------------------

EXAMPLE_GOOD_FEEDBACK = {
    "summary": (
        "Your elbow flare at 68° is 18° outside the optimal range (30–50°), "
        "which pushes the ball off the shooting line and causes the flat arc. "
        "Your jump-to-release timing is solid at 320ms, but the elbow issue is "
        "creating inconsistency in the wrist snap."
    ),
    "primary_correction": (
        "Tuck your shooting elbow under the ball at the point of release — "
        "visualise your elbow pointing at the basket, not sideways."
    ),
    "detailed_analysis": (
        "The right elbow angle of 68° at release significantly exceeds the "
        "NBA-calibrated optimal range of 30–50°. This creates a lateral force "
        "component that deflects the ball from the intended arc. Combined with "
        "the detected alignment score of 71%, the shot has consistent mechanical "
        "compensations. The knee extension (165°) is within optimal range, "
        "indicating the lower body kinetic chain is solid. Focus all corrections "
        "on upper body alignment."
    ),
    "confidence": 0.82,
    "drills": [
        {
            "name": "Wall Elbow Tuck Drill",
            "description": (
                "Stand 30cm from a wall. Shoot with one hand, "
                "ensuring elbow stays tucked and does not contact the wall."
            ),
            "duration_minutes": 10,
            "focus": "Elbow alignment at release",
            "difficulty": "intermediate",
        }
    ],
}
