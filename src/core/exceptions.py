"""Domain exceptions for AI Shoot."""


class ShootAIError(Exception):
    """Base exception for the project."""


class PerceptionError(ShootAIError):
    """Raised when video/pose processing fails."""


class NoPoseDetectedError(PerceptionError):
    """Raised when no player/skeleton is found in the video."""


class AnalysisError(ShootAIError):
    """Raised when biomechanical analysis fails."""


class VLMError(ShootAIError):
    """Raised when the VLM call fails or returns unusable output."""


class AgentError(ShootAIError):
    """Raised when an agent fails to complete its task."""
