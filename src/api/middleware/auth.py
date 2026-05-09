"""X-API-Key authentication middleware.

Enforces API key authentication on all HTTP endpoints and WebSocket connections.

Rules:
- Disabled when ``settings.api_keys`` is empty (dev / CI mode — no key required).
- ``GET /health`` is always exempt (Cloud Run liveness probe cannot send auth headers).
- Returns 401 when the ``X-API-Key`` header is absent.
- Returns 403 when the key is provided but not in ``settings.api_keys``.
- For WebSocket connections, closes with code 4403 (custom app-level forbidden).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send

from src.core.config import settings

# Paths that bypass authentication (no trailing slash variants needed — FastAPI
# normalises paths before they reach the ASGI scope).
_EXEMPT_PATHS: frozenset[str] = frozenset({"/health"})


class APIKeyMiddleware:
    """Pure-ASGI middleware that validates ``X-API-Key`` on every request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Auth disabled when no keys are configured (local dev / CI)
        if not settings.api_keys:
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")

        # Cloud Run liveness probe — always allow
        if scope["type"] == "http" and path in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        headers: dict[bytes, bytes] = dict(scope.get("headers", []))
        raw_key = headers.get(b"x-api-key", b"")
        api_key = raw_key.decode(errors="replace").strip()

        if not api_key:
            await self._reject(scope, send, 401, "Missing X-API-Key header")
            return

        if api_key not in settings.api_keys:
            await self._reject(scope, send, 403, "Invalid API key")
            return

        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, send: Send, status: int, detail: str) -> None:
        if scope["type"] == "websocket":
            # WS handshake hasn't been accepted yet — send close immediately.
            # Code 4403 = custom application-level "forbidden".
            await send({"type": "websocket.close", "code": 4403})
            return

        body = json.dumps({"detail": detail}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
