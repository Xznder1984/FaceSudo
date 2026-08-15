"""SQLite storage for face encodings, encrypted at rest with Fernet.

The Fernet key is stored in the macOS Keychain (never on disk). Each
enrolled sample is one row; the 128-float encoding is pickled and
encrypted before being written to the DB.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import uuid
from datetime import datetime, timezone

from cryptography.fernet import Fernet, InvalidToken

from . import keychain
from .config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS face_samples (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL,
    encoding BLOB NOT NULL
);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Encryption:
    """Loads or creates the Fernet key in Keychain and encrypts/decrypts."""

    def __init__(self) -> None:
        raw = keychain.get_encryption_key()
        if raw is None:
            key = Fernet.generate_key()
            self._fernet = Fernet(key)
            keychain.set_encryption_key(key)
        else:
            try:
                self._fernet = Fernet(raw)
            except (ValueError, TypeError):
                raise RuntimeError(
                    "Encryption key in Keychain is invalid. Re-run `facesudo reset` "
                    "to generate a fresh key (this will erase stored encodings)."
                )

    def encrypt(self, payload: bytes) -> bytes:
        return self._fernet.encrypt(payload)

    def decrypt(self, blob: bytes) -> bytes:
        try:
            return self._fernet.decrypt(blob)
        except InvalidToken:
            raise RuntimeError("Stored encoding could not be decrypted (key mismatch).")


class EncodingStore:
    def __init__(self, db_path=DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._crypto = Encryption()
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def add_sample(self, label: str, encoding: list) -> None:
        """`encoding` is a list/tuple of 128 floats."""
        row_id = str(uuid.uuid4())
        payload = pickle.dumps({"v": 1, "e": [float(x) for x in encoding]})
        blob = self._crypto.encrypt(payload)
        with self._conn:
            self._conn.execute(
                "INSERT INTO face_samples(id, label, created_at, encoding) VALUES (?, ?, ?, ?)",
                (row_id, label, _utcnow(), blob),
            )

    def all(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, created_at, encoding FROM face_samples ORDER BY created_at"
        ).fetchall()
        out = []
        for rid, label, created_at, blob in rows:
            try:
                payload = pickle.loads(self._crypto.decrypt(blob))
                enc = list(payload["e"])
            except Exception:
                enc = []
            out.append({"id": rid, "label": label, "created_at": created_at, "encoding": enc})
        return out

    def encodings(self) -> list[list[float]]:
        return [row["encoding"] for row in self.all() if row["encoding"]]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM face_samples").fetchone()[0]

    def clear(self, label: str | None = None) -> int:
        with self._conn:
            if label is None:
                cur = self._conn.execute("DELETE FROM face_samples")
            else:
                cur = self._conn.execute("DELETE FROM face_samples WHERE label = ?", (label,))
        return cur.rowcount
