"""
Gemini Flash 2.0 VLM client.

Uses the google-generativeai SDK. Supports:
- Text completion (complete)
- JSON-mode completion (complete_json) via response_mime_type
- Configurable safety thresholds (disabled for sports analysis)
"""

from __future__ import annotations

import json
import logging
import random  # noqa: S311 — jitter for retry backoff is not a cryptographic purpose
import time
from typing import TYPE_CHECKING, Any

from src.vlm.base import BaseVLMClient, Message, VLMConfig, VLMError, VLMParseError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# Transient Google API errors that warrant a retry. Authentication, invalid
# argument, or content-policy errors are NOT in this list — retrying them is
# pointless. We import lazily so the module still works when google-api-core
# is not installed (tests / CI without VLM deps).
def _build_retryable_types() -> tuple[type[BaseException], ...]:
    try:
        from google.api_core import exceptions as gax_exc  # noqa: PLC0415

        return (
            gax_exc.DeadlineExceeded,
            gax_exc.ServiceUnavailable,
            gax_exc.InternalServerError,
            gax_exc.ResourceExhausted,  # rate limit / quota
            gax_exc.Aborted,
            gax_exc.GatewayTimeout,
        )
    except ImportError:
        # No SDK installed — the client itself can't be built, but keep the
        # tuple type-stable so _call_with_retry stays simple.
        return (TimeoutError, ConnectionError)


_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = _build_retryable_types()


class GeminiFlashClient(BaseVLMClient):
    """
    Gemini Flash 2.0 client via google-generativeai SDK.

    Import of the SDK is deferred to __init__ so tests can mock it without
    installing google-generativeai in environments without API access.

    Example::

        client = GeminiFlashClient.from_env()
        response = client.complete([Message(role="user", content="Analyse ce tir.")])
    """

    def __init__(self, api_key: str, config: VLMConfig | None = None) -> None:
        super().__init__(config or VLMConfig(model="gemini-2.0-flash"))
        try:
            import google.generativeai as genai  # noqa: PLC0415

            genai.configure(api_key=api_key)
            self._genai = genai
            self._model = genai.GenerativeModel(
                model_name=self.config.model,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                },
                # Disable safety filters that incorrectly flag biomechanics
                # terms like "elbow lock" or "wrist snap" as harmful.
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE",
                    },
                ],
            )
            self._json_model = genai.GenerativeModel(
                model_name=self.config.model,
                generation_config={
                    "temperature": min(self.config.temperature, 0.1),
                    "max_output_tokens": self.config.max_tokens,
                    "response_mime_type": "application/json",
                },
            )
        except ImportError as exc:
            raise VLMError(
                "google-generativeai not installed. Run: uv add google-generativeai"
            ) from exc

    @classmethod
    def from_env(cls, config: VLMConfig | None = None) -> GeminiFlashClient:
        """Build client from GEMINI_API_KEY environment variable."""
        import os

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise VLMError("GEMINI_API_KEY environment variable not set.")
        return cls(api_key=api_key, config=config)

    def _messages_to_gemini(self, messages: list[Message]) -> list[dict]:
        """
        Convert our Message format to Gemini's content list format.

        Gemini uses "user"/"model" roles (not "assistant").
        System messages are prepended to the first user message.
        """
        result = []
        system_prefix = ""
        for msg in messages:
            if msg.role == "system":
                system_prefix += msg.content + "\n\n"
            elif msg.role == "user":
                content = (system_prefix + msg.content) if system_prefix else msg.content
                result.append({"role": "user", "parts": [content]})
                system_prefix = ""
            elif msg.role == "assistant":
                result.append({"role": "model", "parts": [msg.content]})
        return result

    def _call_with_retry(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Invoke ``fn`` with exponential backoff + full jitter on transient errors.

        Retries up to ``self.config.retry_attempts`` times. The base backoff
        doubles each attempt and is capped at ``retry_max_backoff_seconds``;
        the actual sleep is then drawn uniformly from ``[0, base]``. This
        "full jitter" pattern (per AWS Architecture Blog) is the simplest
        decorrelation that prevents thundering-herd retries when many
        instances fail at the same instant (e.g. a regional Gemini quota
        hit). Non-retryable exceptions propagate immediately.
        """
        last_exc: BaseException | None = None
        total_attempts = self.config.retry_attempts + 1
        for attempt in range(total_attempts):
            try:
                return fn(*args, **kwargs)
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt + 1 >= total_attempts:
                    break
                base_backoff = min(
                    self.config.retry_backoff_seconds * (2**attempt),
                    self.config.retry_max_backoff_seconds,
                )
                # Full jitter — uniform on [0, base_backoff]. Decorrelates
                # multi-instance retry storms without sacrificing the average
                # backoff growth.
                sleep_for = random.uniform(0, base_backoff)  # noqa: S311
                logger.warning(
                    "Gemini transient error (attempt %d/%d): %s — retrying in %.2fs (base %.1fs, jittered)",
                    attempt + 1,
                    total_attempts,
                    exc,
                    sleep_for,
                    base_backoff,
                )
                time.sleep(sleep_for)
        # All retries exhausted — raise as VLMError so callers get a stable type.
        assert last_exc is not None  # noqa: S101 — invariant of the loop above
        raise VLMError(
            f"Gemini API error after {total_attempts} attempts: {last_exc}"
        ) from last_exc

    def _request_options(self) -> dict[str, Any]:
        """Per-call request_options dict (timeout etc.) for the SDK."""
        return {"timeout": self.config.timeout_seconds}

    def _dispatch(
        self,
        model: Any,
        gemini_messages: list[dict],
    ) -> Any:
        """
        Single-shot call to Gemini — used as the retry-wrapped unit.

        Takes a *list* (not a one-shot iterator/generator) on purpose: the
        retry layer may call this multiple times, and a generator would be
        exhausted after the first attempt, leaving subsequent retries with
        no input. Callers must materialize their messages before reaching
        this method (which `_messages_to_gemini` already does).
        """
        if len(gemini_messages) == 1:
            return model.generate_content(
                gemini_messages[0]["parts"][0],
                request_options=self._request_options(),
            )
        chat = model.start_chat(history=gemini_messages[:-1])  # type: ignore[arg-type]
        last = gemini_messages[-1]["parts"][0]
        return chat.send_message(last, request_options=self._request_options())

    def complete(self, messages: list[Message]) -> str:
        """Send messages and return text response from Gemini."""
        gemini_messages = self._messages_to_gemini(messages)
        if not gemini_messages:
            raise VLMError("No user messages provided.")
        try:
            response = self._call_with_retry(self._dispatch, self._model, gemini_messages)
            text = response.text
            logger.debug("Gemini response: %d chars", len(text))
            return text
        except VLMError:
            raise
        except Exception as exc:
            # Non-retryable, non-VLM error (auth, invalid arg, content policy)
            raise VLMError(f"Gemini API error: {exc}") from exc

    def complete_json(self, messages: list[Message]) -> dict:
        """Send messages and return parsed JSON from Gemini JSON mode."""
        gemini_messages = self._messages_to_gemini(messages)
        if not gemini_messages:
            raise VLMError("No user messages provided.")
        try:
            response = self._call_with_retry(self._dispatch, self._json_model, gemini_messages)
            # `response.text` itself can raise on safety-blocked completions
            # (the SDK exposes the block reason via attribute access). Keep
            # it inside the try so the VLMError wrapping contract holds.
            raw = response.text.strip()
        except VLMError:
            raise
        except Exception as exc:
            raise VLMError(f"Gemini API error: {exc}") from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as parse_exc:
            raise VLMParseError(f"Gemini returned non-JSON response: {raw[:200]}") from parse_exc
