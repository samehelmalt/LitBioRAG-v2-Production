"""Knowledge-base registry: who owns which uploaded files, and their versions (SQLite)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_DDL = """
CREATE TABLE IF NOT EXISTS kbs (
  kb_id TEXT PRIMARY KEY, name TEXT NOT NULL, owner TEXT NOT NULL, created_at REAL NOT NULL,
  description TEXT, shared_with TEXT NOT NULL DEFAULT '[]', version INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS kb_files (
  kb_id TEXT NOT NULL, file_id TEXT NOT NULL, file_name TEXT NOT NULL, kind TEXT NOT NULL,
  sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, uploaded_by TEXT NOT NULL,
  uploaded_at REAL NOT NULL,
  n_units INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL, error TEXT, path TEXT NOT NULL,
  meta TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (kb_id, file_id)
);
"""


class KBRegistry:
    def __init__(self, path: Path | str | None):
        self._conn = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(kb_files)")}
        if "meta" not in cols:
            with self._conn:
                self._conn.execute(
                    "ALTER TABLE kb_files ADD COLUMN meta TEXT NOT NULL DEFAULT '{}'")
        self._lock = threading.Lock()

    # ---- kbs ------------------------------------------------------------------------------
    def create(self, *, name: str, owner: str, description: str = "") -> dict[str, Any]:
        kb_id = "kb_" + uuid.uuid4().hex[:12]
        with self._lock, self._conn:
            self._conn.execute("INSERT INTO kbs VALUES (?,?,?,?,?,?,0)",
                               (kb_id, name, owner, time.time(), description, "[]"))  # fmt: skip
        return self.get(kb_id)  # type: ignore[return-value]

    def get(self, kb_id: str) -> dict[str, Any] | None:
        r = self._conn.execute("SELECT * FROM kbs WHERE kb_id=?", (kb_id,)).fetchone()
        return _kb(r) if r else None

    def list_for(self, sub: str, *, admin: bool = False) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM kbs ORDER BY created_at DESC").fetchall()
        out = [_kb(r) for r in rows]
        if admin:
            return out
        return [k for k in out if k["owner"] == sub or sub in k["shared_with"]]

    def can_read(self, kb_id: str, sub: str, *, admin: bool = False) -> bool:
        k = self.get(kb_id)
        return bool(k) and (admin or k["owner"] == sub or sub in k["shared_with"])

    def can_write(self, kb_id: str, sub: str, *, admin: bool = False) -> bool:
        k = self.get(kb_id)
        return bool(k) and (admin or k["owner"] == sub)

    def share(self, kb_id: str, subs: list[str]) -> dict[str, Any]:
        with self._lock, self._conn:
            self._conn.execute("UPDATE kbs SET shared_with=? WHERE kb_id=?",
                               (json.dumps(sorted(set(subs))), kb_id))  # fmt: skip
        return self.get(kb_id)  # type: ignore[return-value]

    def bump_version(self, kb_id: str) -> int:
        with self._lock, self._conn:
            self._conn.execute("UPDATE kbs SET version=version+1 WHERE kb_id=?", (kb_id,))
        return self.get(kb_id)["version"]  # type: ignore[index]

    def delete(self, kb_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM kb_files WHERE kb_id=?", (kb_id,))
            self._conn.execute("DELETE FROM kbs WHERE kb_id=?", (kb_id,))

    # ---- files ----------------------------------------------------------------------------
    def upsert_file(self, kb_id: str, *, file_id: str, file_name: str, kind: str, sha256: str,
                    size: int, uploaded_by: str, path: str,
                    status: str = "queued", meta: dict | None = None) -> dict:  # fmt: skip
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO kb_files VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (kb_id, file_id, file_name, kind, sha256, size, uploaded_by, time.time(), 0,
                 status, None, path, json.dumps(meta or {})),
            )
        return self.get_file(kb_id, file_id)  # type: ignore[return-value]

    def set_file_status(self, kb_id: str, file_id: str, status: str, *, n_units: int | None = None,
                        error: str | None = None) -> None:  # fmt: skip
        with self._lock, self._conn:
            if n_units is None:
                self._conn.execute(
                    "UPDATE kb_files SET status=?, error=? WHERE kb_id=? AND file_id=?",
                    (status, error, kb_id, file_id))
            else:
                self._conn.execute(
                    "UPDATE kb_files SET status=?, error=?, n_units=? WHERE kb_id=? AND file_id=?",
                    (status, error, n_units, kb_id, file_id))

    def get_file(self, kb_id: str, file_id: str) -> dict | None:
        r = self._conn.execute("SELECT * FROM kb_files WHERE kb_id=? AND file_id=?",
                               (kb_id, file_id)).fetchone()  # fmt: skip
        return _file(r) if r else None

    def files(self, kb_id: str) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM kb_files WHERE kb_id=? ORDER BY uploaded_at",
                                  (kb_id,)).fetchall()  # fmt: skip
        return [_file(r) for r in rows]

    def set_file_meta(self, kb_id: str, file_id: str, **updates) -> None:
        f = self.get_file(kb_id, file_id)
        if f is None:
            return
        meta = {**f["meta"], **updates}
        with self._lock, self._conn:
            self._conn.execute("UPDATE kb_files SET meta=? WHERE kb_id=? AND file_id=?",
                               (json.dumps(meta), kb_id, file_id))  # fmt: skip

    def previous_versions(self, kb_id: str, course_id: str, lecture_id: str,
                          exclude_file_id: str) -> list[dict]:  # fmt: skip
        if not lecture_id:
            return []
        return [f for f in self.files(kb_id)
                if f["file_id"] != exclude_file_id and f["meta"].get("lecture_id") == lecture_id
                and f["meta"].get("course_id", "") == course_id
                and not f["meta"].get("superseded_by")]  # fmt: skip

    def delete_file(self, kb_id: str, file_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM kb_files WHERE kb_id=? AND file_id=?", (kb_id, file_id))


def _file(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    try:
        d["meta"] = json.loads(d.get("meta") or "{}")
    except ValueError:
        d["meta"] = {}
    return d


def _kb(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    d["shared_with"] = json.loads(d.get("shared_with") or "[]")
    return d
