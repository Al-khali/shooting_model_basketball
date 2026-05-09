"""
Player memory service — persistent session state for personalized coaching.

This module implements the POC in-memory + JSON file store that mirrors the
interface of Google ADK Memory Bank. Swapping to ADK Agent Runtime's Memory
Bank in production is a single-file change.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from src.agents.state import PlayerSession, ShotRecord
from src.api.schemas.domain import CoachingFeedback, PlayerLevel, SessionContext

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = Path("data/players")


class PlayerMemoryService:
    """
    Reads and writes per-player coaching history to JSON files.

    Designed to mirror ADK Memory Bank's interface so that swapping to
    the production service (Agent Runtime) requires zero changes in callers.

    Usage::

        svc = PlayerMemoryService()
        context = svc.build_context("player_42", PlayerLevel.INTERMEDIATE)
        # ... run analysis ...
        svc.record_feedback("player_42", feedback, PlayerLevel.INTERMEDIATE)
    """

    def __init__(self, store_dir: Path | str = DEFAULT_STORE_DIR) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _player_path(self, player_id: str) -> Path:
        safe_id = player_id.replace("/", "_").replace("..", "_")
        return self.store_dir / f"{safe_id}.json"

    def load(self, player_id: str) -> PlayerSession:
        """Load a player session, returning an empty one if none exists."""
        path = self._player_path(player_id)
        if not path.exists():
            logger.info("No history for player %s — creating new session", player_id)
            return PlayerSession(player_id=player_id)
        try:
            data = json.loads(path.read_text())
            return PlayerSession.model_validate(data)
        except Exception as e:
            logger.warning("Could not load player %s history: %s — resetting", player_id, e)
            return PlayerSession(player_id=player_id)

    def save(self, session: PlayerSession) -> None:
        """Persist a player session to disk."""
        path = self._player_path(session.player_id)
        try:
            path.write_text(session.model_dump_json(indent=2))
        except Exception as e:
            logger.error("Could not save player %s history: %s", session.player_id, e)

    def build_context(
        self,
        player_id: str,
        player_level: PlayerLevel,
        notes: str | None = None,
    ) -> SessionContext:
        """Build a SessionContext from stored player history."""
        session = self.load(player_id)
        return SessionContext(
            player_id=player_id,
            player_level=player_level,
            session_count=session.total_sessions,
            recurring_issues=session.recurring_issues,
            previous_drills=session.recent_drills,
            notes=notes,
        )

    def record_feedback(
        self,
        player_id: str,
        feedback: CoachingFeedback,
        player_level: PlayerLevel,
    ) -> PlayerSession:
        """
        Persist a coaching result and update player history.

        Updates issue frequency counts, drill history, and appends a shot record.
        Returns the updated PlayerSession.
        """
        session = self.load(player_id)
        session.player_level = player_level
        session.total_sessions += 1

        # Track issue frequencies
        issues = list(feedback.biomechanics.issues_detected)
        if feedback.biomechanics.primary_issue:
            issues = [feedback.biomechanics.primary_issue] + issues
        for issue in issues:
            session.issue_counts[issue] = session.issue_counts.get(issue, 0) + 1

        # Track drills (append only; deduplication handled in recent_drills)
        for drill in feedback.drills:
            session.drills_history.append(drill.name)

        # Append shot record
        record = ShotRecord(
            session_id=str(uuid4()),
            timestamp=datetime.now(UTC),
            shot_result=feedback.shot_result,
            primary_correction=feedback.primary_correction,
            drills_assigned=[d.name for d in feedback.drills],
            alignment_score=feedback.biomechanics.alignment_score,
        )
        session.shot_history.append(record)

        self.save(session)
        return session
