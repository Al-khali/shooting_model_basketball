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

from src.vlm.base import BaseVLMClient, Message, VLMConfig, VLMError, VLMParseError

logger = logging.getLogger(__name__)


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

    def complete(self, messages: list[Message]) -> str:
        """Send messages and return text response from Gemini."""
        try:
            gemini_messages = self._messages_to_gemini(messages)
            if not gemini_messages:
                raise VLMError("No user messages provided.")
            # Multi-turn: use start_chat + send_message
            if len(gemini_messages) == 1:
                response = self._model.generate_content(gemini_messages[0]["parts"][0])
            else:
                chat = self._model.start_chat(history=gemini_messages[:-1])  # type: ignore[arg-type]
                last = gemini_messages[-1]["parts"][0]
                response = chat.send_message(last)
            text = response.text
            logger.debug("Gemini response: %d chars", len(text))
            return text
        except Exception as exc:
            raise VLMError(f"Gemini API error: {exc}") from exc

    def complete_json(self, messages: list[Message]) -> dict:
        """Send messages and return parsed JSON from Gemini JSON mode."""
        try:
            gemini_messages = self._messages_to_gemini(messages)
            if not gemini_messages:
                raise VLMError("No user messages provided.")
            if len(gemini_messages) == 1:
                response = self._json_model.generate_content(gemini_messages[0]["parts"][0])
            else:
                chat = self._json_model.start_chat(history=gemini_messages[:-1])  # type: ignore[arg-type]
                last = gemini_messages[-1]["parts"][0]
                response = chat.send_message(last)
            raw = response.text.strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError as parse_exc:
                raise VLMParseError(
                    f"Gemini returned non-JSON response: {raw[:200]}"
                ) from parse_exc
        except VLMParseError:
            raise
        except Exception as exc:
            raise VLMError(f"Gemini API error: {exc}") from exc
