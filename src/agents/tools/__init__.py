"""Tool functions for the AI Shoot agent pipeline.

Each function is a plain Python callable that can be decorated with
``@google.adk.tools.tool`` at runtime in the ADK pipeline factory.
They work as regular functions in tests and non-ADK contexts.
"""

from src.agents.tools.analysis_tools import compute_biomechanics
from src.agents.tools.coach_tools import generate_coaching_feedback
from src.agents.tools.memory_tools import load_player_history, save_coaching_result
from src.agents.tools.perception_tools import extract_shot_frames
from src.agents.tools.planner_tools import build_training_plan

__all__ = [
    "extract_shot_frames",
    "compute_biomechanics",
    "generate_coaching_feedback",
    "build_training_plan",
    "load_player_history",
    "save_coaching_result",
]
