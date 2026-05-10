"""
Abstract VLM client interface.

All VLM backends (Gemini, OpenAI, Qwen2-VL, local Ollama) must implement
this contract so the rest of the pipeline stays model-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Message:
    """Single message in a conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class VLMConfig:
    """Runtime configuration for a VLM backend."""

    model: str = "gemini-2.0-flash"
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout_seconds: float = 30.0
    # Resilience: retry on transient errors (network blips, rate limits, 5xx).
    # Total attempts = 1 + retry_attempts. With backoff = 1.0s and 3 retries,
    # the worst-case wall time is ~1+2+4+8 = 15s of waits + 4 timeouts.
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    retry_max_backoff_seconds: float = 16.0
    # Model-specific extras (e.g. safety settings)
    extra: dict = field(default_factory=dict)


class BaseVLMClient(ABC):
    """
    Minimal VLM client contract.

    Implementations must be stateless — one client instance per configuration,
    no session state stored here.
    """

    def __init__(self, config: VLMConfig) -> None:
        self.config = config

    @abstractmethod
    def complete(self, messages: list[Message]) -> str:
        """
        Send messages and return the text response.

        Args:
            messages: Ordered list of conversation messages.

        Returns:
            Model response as a plain string.

        Raises:
            VLMError: On API errors, timeouts, or content policy rejections.
        """

    @abstractmethod
    def complete_json(self, messages: list[Message]) -> dict:
        """
        Send messages expecting a JSON response.

        The implementation must parse the model output into a Python dict.
        Use this for structured coaching feedback to avoid brittle string parsing.

        Args:
            messages: Ordered list of conversation messages.

        Returns:
            Parsed JSON as a dict.

        Raises:
            VLMError: On API errors.
            VLMParseError: If the model response is not valid JSON.
        """

    @property
    def model_id(self) -> str:
        """Return the model identifier string for metadata."""
        return self.config.model


class VLMError(Exception):
    """Raised when the VLM backend returns an error."""


class VLMParseError(VLMError):
    """Raised when the model response cannot be parsed as expected."""
