"""
LiteLLM-backed VLM client with automatic provider failover.

Implements ``BaseVLMClient`` via the ``litellm`` library in-process. The same
API surface as a future LiteLLM Proxy sidecar (ADR-001 D3): swapping from
in-process to a proxy URL is a one-line constructor change, no caller code
moves.

Why LiteLLM instead of provider-specific clients (``google.generativeai``,
``openai``, ``anthropic``)?

- **Failover** : if the primary provider returns a retryable error or hits a
  hard quota, LiteLLM automatically tries the configured fallback models.
  No bespoke chain logic in caller code.
- **Provider portability** : the same ``messages`` format works for Gemini,
  Claude, GPT, Qwen2-VL — the call site doesn't change when the provider
  changes.
- **Cost tracking** : LiteLLM emits standard usage events that can be
  aggregated cross-provider (relevant for ADR-001 R3 risk mitigation).

POC zero-budget posture: the default config uses only Gemini (matches the
behaviour of ``GeminiFlashClient``). Fallbacks are opt-in via the
``LITELLM_FALLBACK_MODELS`` env var (comma-separated LiteLLM model strings)
and require the respective provider API keys (``ANTHROPIC_API_KEY``,
``OPENAI_API_KEY``) to be set in the environment — when unset, LiteLLM
skips that fallback rather than erroring on the primary call.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from src.vlm.base import BaseVLMClient, Message, VLMConfig, VLMError, VLMParseError

logger = logging.getLogger(__name__)

# Default primary model. LiteLLM uses ``<provider>/<model>`` notation;
# ``gemini/gemini-2.0-flash`` is the equivalent of the GeminiFlashClient
# default and matches the value used elsewhere in the project.
DEFAULT_PRIMARY_MODEL = "gemini/gemini-2.0-flash"


class LiteLLMClient(BaseVLMClient):
    """
    Provider-agnostic VLM client with optional automatic failover.

    Example::

        client = LiteLLMClient.from_env()  # POC: Gemini-only by default
        text = client.complete([Message(role="user", content="Analyse ce tir.")])

    To activate failover in production, set in the runtime environment::

        export LITELLM_FALLBACK_MODELS="claude-3-5-haiku-latest,openai/gpt-4o-mini"
        export ANTHROPIC_API_KEY=...
        export OPENAI_API_KEY=...

    The constructor does not validate fallback keys — if a fallback model's
    key is missing, LiteLLM skips that step. The primary model's key
    (``GEMINI_API_KEY`` for the default) is still required.
    """

    def __init__(
        self,
        primary_model: str = DEFAULT_PRIMARY_MODEL,
        fallback_models: list[str] | None = None,
        config: VLMConfig | None = None,
    ) -> None:
        super().__init__(config or VLMConfig(model=primary_model))
        self._primary_model = primary_model
        self._fallback_models = fallback_models or []

        try:
            import litellm  # noqa: PLC0415

            self._litellm = litellm
        except ImportError as exc:
            raise VLMError("litellm not installed. Run: uv add litellm") from exc

        # LiteLLM honours num_retries and timeout via the call args, but
        # surfacing the config defaults here makes the wiring explicit and
        # keeps callers from having to know the LiteLLM-specific knobs.
        logger.debug(
            "LiteLLMClient initialised: primary=%s fallbacks=%s",
            self._primary_model,
            self._fallback_models or "(none)",
        )

    @classmethod
    def from_env(cls, config: VLMConfig | None = None) -> LiteLLMClient:
        """
        Build a client from environment variables.

        ``LITELLM_PRIMARY_MODEL`` overrides the default primary (rarely needed).
        ``LITELLM_FALLBACK_MODELS`` (CSV) declares the failover chain. Missing
        provider API keys cause the corresponding fallback to be skipped at
        runtime, not at construction time.
        """
        primary = os.environ.get("LITELLM_PRIMARY_MODEL", DEFAULT_PRIMARY_MODEL)
        fallbacks_raw = os.environ.get("LITELLM_FALLBACK_MODELS", "").strip()
        fallbacks = [m.strip() for m in fallbacks_raw.split(",") if m.strip()]

        # Sanity check the primary provider's key. We only check the default
        # provider (Gemini) — custom primaries get the LiteLLM default error
        # if mis-configured.
        if primary.startswith("gemini/") and not os.environ.get("GEMINI_API_KEY"):
            raise VLMError(
                f"GEMINI_API_KEY environment variable not set "
                f"(required for primary model: {primary})"
            )

        return cls(
            primary_model=primary,
            fallback_models=fallbacks,
            config=config,
        )

    def _call(self, messages: list[Message], *, json_mode: bool = False) -> Any:
        """
        Wrapper around ``litellm.completion`` that maps errors and applies
        fallbacks. Returns the raw LiteLLM ``ModelResponse`` object so
        callers can decide how to extract the content.
        """
        request_kwargs: dict[str, Any] = {
            "model": self._primary_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout_seconds,
            "num_retries": self.config.retry_attempts,
        }
        if self._fallback_models:
            request_kwargs["fallbacks"] = self._fallback_models
        if json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}

        try:
            return self._litellm.completion(**request_kwargs)
        except Exception as exc:
            # LiteLLM raises provider-agnostic exceptions
            # (``litellm.exceptions.*``). We don't import them explicitly to
            # avoid coupling the test surface to LiteLLM internals — the
            # contract here is: any failure surfaces as ``VLMError`` with
            # a stable type name in the message for log triage.
            raise VLMError(f"litellm_call_failed:{type(exc).__name__}") from exc

    def complete(self, messages: list[Message]) -> str:
        """Send messages and return the text response."""
        if not messages:
            raise VLMError("No user messages provided.")
        response = self._call(messages, json_mode=False)
        try:
            text = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise VLMError(f"litellm_response_malformed:{type(exc).__name__}") from exc
        if not isinstance(text, str):
            raise VLMError("litellm_response_not_text")
        logger.debug("LiteLLM response: %d chars (model=%s)", len(text), self._primary_model)
        return text

    def complete_json(self, messages: list[Message]) -> dict:
        """Send messages and return parsed JSON."""
        if not messages:
            raise VLMError("No user messages provided.")
        response = self._call(messages, json_mode=True)
        try:
            raw = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise VLMError(f"litellm_response_malformed:{type(exc).__name__}") from exc
        if not isinstance(raw, str):
            raise VLMError("litellm_response_not_text")
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError as parse_exc:
            raise VLMParseError(f"LiteLLM returned non-JSON response: {raw[:200]}") from parse_exc
