"""Tool function: build a personalized weekly training plan.

Pure Python — no LLM or heavy dependencies required.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_training_plan(
    feedback_json: str,
    session_count: int = 0,
    recurring_issues_json: str = "[]",
    previous_drills_json: str = "[]",
) -> dict[str, Any]:
    """
    Build a personalized weekly training plan from coaching feedback.

    Adapts the plan based on session history:
    - Avoids repeating recently assigned drills
    - Escalates recurring issues to top priority
    - Increases weekly session count as the player progresses

    Args:
        feedback_json: JSON string matching CoachingFeedback schema.
        session_count: Number of completed sessions (0 = first time).
        recurring_issues_json: JSON list of recurring issues (most frequent first).
        previous_drills_json: JSON list of previously assigned drill names.

    Returns:
        Training plan dict with keys:
        {focus_area, weekly_sessions, intensity, drills,
         primary_correction, progression_note, session_number}.
        Includes "error" key on parse failure.
    """
    try:
        feedback_data = json.loads(feedback_json)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid feedback JSON: {e}"}

    try:
        recurring_issues: list[str] = json.loads(recurring_issues_json)
    except json.JSONDecodeError:
        recurring_issues = []

    try:
        previous_drills: list[str] = json.loads(previous_drills_json)
    except json.JSONDecodeError:
        previous_drills = []

    drills_raw: list[dict[str, Any]] = feedback_data.get("drills") or []

    # Prefer drills not yet seen; fall back to all drills if all are repeats
    new_drills = [d for d in drills_raw if d.get("name") not in previous_drills]
    selected_drills = new_drills or drills_raw

    # Increase volume as player progresses
    if session_count == 0:
        weekly_sessions = 3
        intensity = "light"
    elif session_count < 5:
        weekly_sessions = 4
        intensity = "moderate"
    else:
        weekly_sessions = 5
        intensity = "progressive"

    # Recurring issues take priority over latest feedback
    if recurring_issues:
        focus_area = f"Persistent issue: {recurring_issues[0]}"
    else:
        focus_area = feedback_data.get("primary_correction", "General technique improvement")

    # Contextual progression note
    if recurring_issues:
        progression_note = (
            "This issue has appeared in previous sessions. "
            "Increase drill repetitions by 20% compared to last session."
        )
    elif session_count > 0:
        progression_note = (
            f"Session {session_count + 1}: Build on previous work. "
            f"Focus on consistency and muscle memory."
        )
    else:
        progression_note = "First session: prioritise form over speed. Quality reps only."

    return {
        "focus_area": focus_area,
        "weekly_sessions": weekly_sessions,
        "intensity": intensity,
        "drills": selected_drills,
        "primary_correction": feedback_data.get("primary_correction", ""),
        "progression_note": progression_note,
        "session_number": session_count + 1,
    }
