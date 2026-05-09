"""In-memory task registry for background analysis jobs.

``TaskStore`` is a module-level singleton that tracks the lifecycle of every
``POST /analyze`` background task. It is intentionally lightweight — no Redis,
no database — which is appropriate for a single-process deployment.

For multi-process or horizontally-scaled deployments, swap this singleton
for a Redis-backed implementation with the same interface.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4


@dataclass
class TaskRecord:
    """Lifecycle record for a single analysis job."""

    task_id: str
    status: Literal["processing", "done", "error"]
    player_id: str | None
    created_at: datetime
    updated_at: datetime
    result: dict[str, Any] | None = field(default=None)
    error: str | None = field(default=None)


class TaskStore:
    """Thread-safe in-memory store for analysis task records."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, player_id: str | None = None) -> TaskRecord:
        """Register a new task and return its record."""
        now = datetime.now(UTC)
        task = TaskRecord(
            task_id=str(uuid4()),
            status="processing",
            player_id=player_id,
            created_at=now,
            updated_at=now,
        )
        async with self._lock:
            self._tasks[task.task_id] = task
        return task

    async def get(self, task_id: str) -> TaskRecord | None:
        """Return the task record or None if not found."""
        async with self._lock:
            return self._tasks.get(task_id)

    async def update(
        self,
        task_id: str,
        *,
        status: Literal["done", "error"],
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Transition a task to its terminal state."""
        async with self._lock:
            if task := self._tasks.get(task_id):
                task.status = status
                task.result = result
                task.error = error
                task.updated_at = datetime.now(UTC)

    async def reset(self) -> None:
        """Clear all tasks — intended for tests only."""
        async with self._lock:
            self._tasks.clear()


# Module-level singleton shared across all routes
task_store = TaskStore()
