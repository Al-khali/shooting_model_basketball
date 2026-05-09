"""
VLM module — AI Shoot Phase 2.

Provides:
- BaseVLMClient: abstract interface for VLM backends
- GeminiFlashClient: Gemini 2.0 Flash implementation
- BasketballVLMAnalyzer: biomechanics → CoachingFeedback
- FeedbackEvaluator: heuristic quality evaluation
- Prompt templates for basketball coaching
"""

from src.vlm.base import BaseVLMClient, Message, VLMConfig, VLMError, VLMParseError
from src.vlm.basketball_analyzer import BasketballVLMAnalyzer
from src.vlm.evaluator import EvaluationResult, FeedbackEvaluator

__all__ = [
    "BaseVLMClient",
    "BasketballVLMAnalyzer",
    "EvaluationResult",
    "FeedbackEvaluator",
    "Message",
    "VLMConfig",
    "VLMError",
    "VLMParseError",
]
