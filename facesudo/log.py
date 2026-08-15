"""Local match-attempt log. Timestamps + outcome only; no images, no
recognizable payloads, no external calls.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS match_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    success INTEGER NOT NULL,
    reason TEXT NOT NULL,
    details TEXT
);
"""


class MatchLog:
    def __init__(self, db_path=DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def append(self, success: bool, reason: str, details: str = "") -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                "INSERT INTO match_log(ts, success, reason, details) VALUES (?, ?, ?, ?)",
                (ts, 1 if success else 0, reason, details),
            )

    def recent(self, limit: int = 200) -> list[dict]:
        rows = self._conn.execute(
            "SELECT ts, success, reason, details FROM match_log "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "ts": ts,
                "success": bool(success),
                "reason": reason,
                "details": details,
            }
            for ts, success, reason, details in rows
        ]

    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM match_log")
