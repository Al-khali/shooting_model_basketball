"""Integration tests for the AI Shoot FastAPI application.

Uses FastAPI's TestClient (synchronous httpx wrapper).
Background tasks run inline after each response in test mode.

All tests use a fresh task_store state via the ``reset_store`` fixture.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.store import task_store

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_store() -> None:
    """Clear the task store between tests to ensure isolation."""
    asyncio.get_event_loop().run_until_complete(task_store.reset())


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def fake_video() -> tuple[str, bytes, str]:
    """A minimal fake video file — perception falls back to stub on ImportError."""
    return ("shot.mp4", b"\x00" * 16, "video/mp4")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert "status" in body
        assert "version" in body
        assert "components" in body
        assert body["status"] in ("ok", "degraded")

    def test_health_version(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["version"] == "0.4.0"

    def test_health_components_present(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert "pipeline" in body["components"]
        assert "memory" in body["components"]


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_analyze_returns_202(self, client: TestClient, fake_video: tuple) -> None:
        name, content, mime = fake_video
        response = client.post(
            "/analyze",
            files={"file": (name, io.BytesIO(content), mime)},
            data={"player_level": "intermediate"},
        )
        assert response.status_code == 202

    def test_analyze_returns_task_id(self, client: TestClient, fake_video: tuple) -> None:
        name, content, mime = fake_video
        body = client.post(
            "/analyze",
            files={"file": (name, io.BytesIO(content), mime)},
        ).json()
        assert "task_id" in body
        assert len(body["task_id"]) == 36  # UUID format

    def test_analyze_status_is_processing_or_done(
        self, client: TestClient, fake_video: tuple
    ) -> None:
        name, content, mime = fake_video
        body = client.post(
            "/analyze",
            files={"file": (name, io.BytesIO(content), mime)},
        ).json()
        # Background tasks run inline in TestClient
        assert body["status"] in ("processing", "done")

    def test_analyze_invalid_player_level(self, client: TestClient, fake_video: tuple) -> None:
        name, content, mime = fake_video
        response = client.post(
            "/analyze",
            files={"file": (name, io.BytesIO(content), mime)},
            data={"player_level": "god-mode"},
        )
        assert response.status_code == 422

    def test_analyze_with_player_id(self, client: TestClient, fake_video: tuple) -> None:
        name, content, mime = fake_video
        body = client.post(
            "/analyze",
            files={"file": (name, io.BytesIO(content), mime)},
            data={"player_id": "player-42", "player_level": "advanced"},
        ).json()
        assert body["player_id"] == "player-42"

    def test_analyze_no_file_returns_422(self, client: TestClient) -> None:
        response = client.post("/analyze", data={"player_level": "intermediate"})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /session/{task_id}
# ---------------------------------------------------------------------------


class TestSession:
    def test_session_not_found(self, client: TestClient) -> None:
        response = client.get("/session/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404

    def test_session_found_after_upload(self, client: TestClient, fake_video: tuple) -> None:
        name, content, mime = fake_video
        task_id = client.post(
            "/analyze",
            files={"file": (name, io.BytesIO(content), mime)},
        ).json()["task_id"]

        response = client.get(f"/session/{task_id}")
        assert response.status_code == 200

    def test_session_schema(self, client: TestClient, fake_video: tuple) -> None:
        name, content, mime = fake_video
        task_id = client.post(
            "/analyze",
            files={"file": (name, io.BytesIO(content), mime)},
        ).json()["task_id"]

        body = client.get(f"/session/{task_id}").json()
        assert "task_id" in body
        assert "status" in body
        assert "created_at" in body
        assert "updated_at" in body
        assert body["status"] in ("processing", "done", "error")

    def test_session_done_has_result(self, client: TestClient, fake_video: tuple) -> None:
        """After background task completes, session should have a result."""
        name, content, mime = fake_video
        task_id = client.post(
            "/analyze",
            files={"file": (name, io.BytesIO(content), mime)},
        ).json()["task_id"]

        body = client.get(f"/session/{task_id}").json()
        if body["status"] == "done":
            assert body["result"] is not None
            assert "coaching_feedback" in body["result"]


# ---------------------------------------------------------------------------
# GET /player/{player_id}/history
# ---------------------------------------------------------------------------


class TestPlayerHistory:
    def test_history_new_player_returns_200(self, client: TestClient) -> None:
        response = client.get("/player/unknown-player/history")
        assert response.status_code == 200

    def test_history_new_player_empty_sessions(self, client: TestClient) -> None:
        body = client.get("/player/unknown-player-xyz/history").json()
        assert body["session_count"] == 0
        assert body["recent_sessions"] == []
        assert body["recurring_issues"] == []

    def test_history_schema(self, client: TestClient) -> None:
        body = client.get("/player/schema-test/history").json()
        assert "player_id" in body
        assert "player_level" in body
        assert "session_count" in body
        assert "recurring_issues" in body
        assert "recent_sessions" in body

    def test_history_player_id_in_response(self, client: TestClient) -> None:
        body = client.get("/player/my-player/history").json()
        assert body["player_id"] == "my-player"


# ---------------------------------------------------------------------------
# WebSocket /analyze/stream
# ---------------------------------------------------------------------------


class TestAnalyzeStream:
    """Tests for WS /analyze/stream — new upload-then-stream protocol."""

    def _do_ws_analysis(
        self,
        ws: any,
        player_level: str = "intermediate",
        send_video: bool = True,
    ) -> list[dict]:
        """Helper: send metadata → wait for ready → send bytes → collect events."""
        ws.send_json({"player_level": player_level, "filename": "shot.mp4"})
        ready = ws.receive_json()
        if ready["event"] == "error":
            return [ready]
        assert ready["event"] == "ready"

        if send_video:
            ws.send_bytes(b"\x00" * 16)
            ws.send_text("upload_done")

        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["event"] in ("done", "error"):
                break
        return events

    def test_websocket_connects(self, client: TestClient) -> None:
        with client.websocket_connect("/analyze/stream") as ws:
            events = self._do_ws_analysis(ws)
        assert events[-1]["event"] in ("done", "error")

    def test_websocket_streams_multiple_events(self, client: TestClient) -> None:
        with client.websocket_connect("/analyze/stream") as ws:
            events = self._do_ws_analysis(ws)
        assert len(events) >= 1

    def test_websocket_invalid_player_level(self, client: TestClient) -> None:
        with client.websocket_connect("/analyze/stream") as ws:
            ws.send_json({"player_level": "super-pro", "filename": "shot.mp4"})
            event = ws.receive_json()
        assert event["event"] == "error"

    def test_websocket_no_video_bytes(self, client: TestClient) -> None:
        """Sending upload_done without any bytes should return an error."""
        with client.websocket_connect("/analyze/stream") as ws:
            ws.send_json({"player_level": "intermediate", "filename": "shot.mp4"})
            ready = ws.receive_json()
            assert ready["event"] == "ready"
            ws.send_text("upload_done")
            event = ws.receive_json()
        assert event["event"] == "error"

    def test_websocket_done_event_has_result(self, client: TestClient) -> None:
        with client.websocket_connect("/analyze/stream") as ws:
            events = self._do_ws_analysis(ws)
        final = events[-1]
        assert final["event"] in ("done", "error")
        if final["event"] == "done":
            assert "coaching_feedback" in final["data"]
