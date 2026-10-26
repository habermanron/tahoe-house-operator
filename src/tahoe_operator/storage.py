import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

from tahoe_operator.config import get_settings


class HouseStore:
    """Small SQLite event store owned by the House MCP service."""

    def __init__(self, database_path: Path | None = None):
        settings = get_settings()
        self.database_path = database_path or settings.database_path
        self.fixture_path = settings.fixture_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS calendar_events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details_json TEXT NOT NULL
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    source_rule_id TEXT NOT NULL
                )"""
            )
            count = db.execute("SELECT COUNT(*) FROM calendar_events").fetchone()[0]
            if count == 0:
                fixture = self.fixture_path / "calendar.json"
                if fixture.exists():
                    for event in json.loads(fixture.read_text()):
                        db.execute(
                            "INSERT INTO calendar_events VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                event["id"],
                                event["event_type"],
                                event["start_date"],
                                event["end_date"],
                                event.get("status", "confirmed"),
                                json.dumps(event.get("details", {})),
                            ),
                        )

    def list_events(self, start: date, end: date) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT * FROM calendar_events
                   WHERE start_date < ? AND end_date > ?
                     AND status IN ('confirmed', 'pending')
                   ORDER BY start_date, event_type""",
                (end.isoformat(), start.isoformat()),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def all_events(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM calendar_events ORDER BY start_date, event_type"
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def create_event_proposal(
        self,
        event_type: str,
        start: date,
        end: date,
        details: dict[str, Any],
        approved: bool,
    ) -> dict[str, Any]:
        if end <= start:
            return {"status": "rejected", "reason": "end_date must be after start_date"}
        if not approved:
            return {
                "status": "needs_approval",
                "proposal_id": str(uuid4()),
                "event_type": event_type,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "details": details,
                "calendar_changed": False,
            }
        conflicts = self.list_events(start, end)
        if conflicts:
            return {"status": "conflict", "conflicts": conflicts, "calendar_changed": False}
        event_id = str(uuid4())
        with self._connect() as db:
            db.execute(
                "INSERT INTO calendar_events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    event_type,
                    start.isoformat(),
                    end.isoformat(),
                    "confirmed",
                    json.dumps(details),
                ),
            )
        return {"status": "created", "event_id": event_id, "calendar_changed": True}

    def log_incident(self, category: str, description: str, source_rule_id: str) -> dict[str, Any]:
        incident_id = str(uuid4())
        with self._connect() as db:
            db.execute(
                "INSERT INTO incidents (id, category, description, source_rule_id) "
                "VALUES (?, ?, ?, ?)",
                (incident_id, category, description, source_rule_id),
            )
        return {"status": "logged", "incident_id": incident_id, "source_rule_id": source_rule_id}

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "event_type": row["event_type"],
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "status": row["status"],
            "details": json.loads(row["details_json"]),
        }
