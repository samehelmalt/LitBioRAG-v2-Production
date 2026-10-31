"""Per-user API keys (hashed at rest). A key acts as its owner (same entitlements, no roles)."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_DDL = """
CREATE TABLE IF NOT EXISTS api_keys (
  key_hash TEXT PRIMARY KEY, prefix TEXT NOT NULL, sub TEXT NOT NULL, name TEXT NOT NULL,
  created_at REAL NOT NULL, last_used_at REAL, revoked_at REAL
);
CREATE INDEX IF NOT EXISTS ix_keys_sub ON api_keys (sub);
"""


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class KeyStore:
    def __init__(self, path: Path | str | None):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._lock = threading.Lock()

    def issue(self, sub: str, name: str) -> dict[str, Any]:
        key = "lbr_" + secrets.token_urlsafe(32)
        with self._lock, self._conn:
            self._conn.execute("INSERT INTO api_keys VALUES (?,?,?,?,?,NULL,NULL)",
                               (_hash(key), key[:10], sub, name[:80], time.time()))  # fmt: skip
        return {"key": key, "prefix": key[:10], "name": name, "sub": sub}

    def lookup(self, key: str) -> str | None:
        """Owner sub for a live key, else None. Updates last_used_at."""
        h = _hash(key)
        r = self._conn.execute("SELECT sub, revoked_at FROM api_keys WHERE key_hash=?",
                               (h,)).fetchone()  # fmt: skip
        if r is None or r["revoked_at"] is not None:
            return None
        with self._lock, self._conn:
            self._conn.execute("UPDATE api_keys SET last_used_at=? WHERE key_hash=?",
                               (time.time(), h))  # fmt: skip
        return r["sub"]

    def list_for(self, sub: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT prefix, name, created_at, last_used_at, revoked_at FROM api_keys WHERE sub=? "
            "ORDER BY created_at DESC", (sub,))
        return [dict(r) for r in rows]

    def revoke(self, sub: str, prefix: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE api_keys SET revoked_at=? WHERE sub=? AND prefix=? AND revoked_at IS NULL",
                (time.time(), sub, prefix))
        return cur.rowcount > 0
