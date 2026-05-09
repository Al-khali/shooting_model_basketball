"""Analysis routes — POST /analyze, GET /session/{task_id}, WS /analyze/stream.

POST /analyze
    Accepts a video file upload (multipart/form-data) + optional player metadata.
    Saves the file to a temp path, registers a task, launches a background job,
    and returns 202 immediately with the task_id for polling.

GET /session/{task_id}
    Poll the background task status. Returns processing / done / error.

WS /analyze/stream
    WebSocket endpoint that accepts JSON params on connect and streams
    pipeline progress events in real time as the analysis runs.
    Event sequence: context_loaded → perception_done → biomechanics_done
                    → coaching_done → plan_done → done (or error)
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

import anyio
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.websockets import WebSocketDisconnect

from src.agents.orchestrator import ShotAnalysisPipeline
from src.api.schemas.domain import PlayerLevel, VideoInput
from src.api.schemas.responses import SessionResponse, TaskResponse
from src.api.store import task_store

router = APIRouter(tags=["analysis"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background task helper
# ---------------------------------------------------------------------------


async def _run_analysis(
    task_id: str,
    video_path: str,
    player_id: str | None,
    player_level: PlayerLevel,
    notes: str | None,
) -> None:
    """Run the sync pipeline in a thread pool and update the task store."""
    try:
        video_input = VideoInput(
            video_path=video_path,
            player_id=player_id,
            player_level=player_level,
            notes=notes,
        )
        pipeline = ShotAnalysisPipeline()

        def _run() -> dict[str, Any]:
            return pipeline.analyze(video_input)

        result = await anyio.to_thread.run_sync(_run)

        if "error" in result:
            await task_store.update(task_id, status="error", error=result["error"])
        else:
            await task_store.update(task_id, status="done", result=result)

    except Exception as exc:
        logger.exception("Background analysis failed for task %s", task_id)
        await task_store.update(task_id, status="error", error=str(exc))
    finally:
        Path(video_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=TaskResponse,
    status_code=202,
    summary="Submit a video for analysis",
)
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file (mp4, mov, avi…)"),  # noqa: B008
    player_id: str | None = Form(None, description="Optional player identifier"),  # noqa: B008
    player_level: str = Form(
        "intermediate", description="beginner / intermediate / advanced / elite"
    ),  # noqa: B008
    notes: str | None = Form(None, description="Optional coach or player notes"),  # noqa: B008
) -> TaskResponse:
    """
    Upload a basketball shot video for asynchronous AI analysis.

    Returns 202 with a ``task_id``. Poll ``GET /session/{task_id}`` for the result.
    """
    try:
        level = PlayerLevel(player_level)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid player_level '{player_level}'. "
            f"Valid values: {[e.value for e in PlayerLevel]}",
        ) from exc

    content = bytearray()
    chunk_size = 1024 * 1024  # 1 MB
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        content.extend(chunk)
    suffix = Path(file.filename).suffix if file.filename else ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    task = await task_store.create(player_id=player_id)

    background_tasks.add_task(
        _run_analysis,
        task_id=task.task_id,
        video_path=tmp_path,
        player_id=player_id,
        player_level=level,
        notes=notes,
    )

    return TaskResponse(
        task_id=task.task_id,
        status="processing",
        player_id=player_id,
        created_at=task.created_at,
    )


@router.get(
    "/session/{task_id}",
    response_model=SessionResponse,
    summary="Poll analysis task status",
)
async def get_session(task_id: str) -> SessionResponse:
    """
    Return the current state of an analysis task.

    - ``status=processing`` — still running, poll again
    - ``status=done`` — ``result`` contains the full pipeline output
    - ``status=error`` — ``error`` contains the failure reason
    """
    task = await task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Session '{task_id}' not found")

    processing_time_ms: float | None = None
    if task.result:
        processing_time_ms = task.result.get("processing_time_ms")

    return SessionResponse(
        task_id=task.task_id,
        status=task.status,
        player_id=task.player_id,
        created_at=task.created_at,
        updated_at=task.updated_at,
        result=task.result,
        error=task.error,
        processing_time_ms=processing_time_ms,
    )


@router.websocket("/analyze/stream")
async def analyze_stream(websocket: WebSocket) -> None:
    """
    Stream analysis progress events over WebSocket.

    Protocol
    --------
    1. Client connects and sends a JSON message with metadata::

        {"player_id": "p42", "player_level": "intermediate", "notes": null,
         "filename": "shot.mp4"}

    2. Server acknowledges::

        {"event": "ready", "data": {}}

    3. Client sends the video file as **binary** frames (one or more chunks).
       Signal end-of-upload with a UTF-8 text message ``"upload_done"``.

    4. Server streams pipeline progress events::

        {"event": "context_loaded",    "data": {"session_count": 2}}
        {"event": "perception_done",   "data": {"player_detected": true, ...}}
        {"event": "biomechanics_done", "data": {"primary_issue": "elbow flare"}}
        {"event": "coaching_done",     "data": {"summary": "...", ...}}
        {"event": "plan_done",         "data": {"drills_count": 2}}
        {"event": "done",              "data": {<full pipeline result>}}

    On error::

        {"event": "error", "data": {"step": "...", "message": "..."}}

    Notes
    -----
    The video is streamed from the client as raw bytes — the server never
    accepts a server-side file path from the client, preventing path traversal.
    """
    await websocket.accept()

    # ---- Step 1: receive metadata JSON ----
    try:
        params = await websocket.receive_json()
    except Exception:
        await websocket.send_json({"event": "error", "data": {"message": "Invalid JSON params"}})
        await websocket.close(code=1003)
        return

    try:
        level = PlayerLevel(params.get("player_level", "intermediate"))
    except ValueError:
        await websocket.send_json({"event": "error", "data": {"message": "Invalid player_level"}})
        await websocket.close()
        return

    filename: str = params.get("filename") or "upload.mp4"
    suffix = Path(filename).suffix or ".mp4"

    await websocket.send_json({"event": "ready", "data": {}})

    # ---- Step 2: receive video bytes ----
    video_chunks: list[bytes] = []
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                if message.get("bytes"):
                    video_chunks.append(message["bytes"])
                elif message.get("text") == "upload_done":
                    break
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.send_json({"event": "error", "data": {"message": "Upload failed"}})
        await websocket.close(code=1011)
        return

    if not video_chunks:
        await websocket.send_json({"event": "error", "data": {"message": "No video data received"}})
        await websocket.close()
        return

    # ---- Step 3: save to temp file (server-controlled path) ----
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        for chunk in video_chunks:
            tmp.write(chunk)
        tmp_path = tmp.name

    video_input = VideoInput(
        video_path=tmp_path,
        player_id=params.get("player_id"),
        player_level=level,
        notes=params.get("notes"),
    )

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def progress_callback(event_name: str, event_data: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"event": event_name, "data": event_data})

    pipeline = ShotAnalysisPipeline()

    async def run_pipeline() -> None:
        def _run() -> dict[str, Any]:
            return pipeline.analyze(video_input, progress_callback=progress_callback)

        try:
            result = await anyio.to_thread.run_sync(_run)
            queue.put_nowait({"event": "done", "data": result})
        except Exception as exc:
            queue.put_nowait({"event": "error", "data": {"message": str(exc)}})
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    pipeline_task = asyncio.create_task(run_pipeline())

    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
            if event["event"] in ("done", "error"):
                break
    except WebSocketDisconnect:
        pipeline_task.cancel()
        return

    await pipeline_task
    await websocket.close()
