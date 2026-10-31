"""Chats: a titled sequence of questions and their request ids (SQLite). Earlier questions are
passed to the generator as context only; evidence is retrieved per question."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_DDL = """
CREATE TABLE IF NOT EXISTS chats (
  chat_id TEXT PRIMARY KEY, sub TEXT NOT NULL, title TEXT NOT NULL,
  created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chats_sub ON chats (sub, updated_at);
CREATE TABLE IF NOT EXISTS messages (
  chat_id TEXT NOT NULL, seq INTEGER NOT NULL, request_id TEXT NOT NULL, question TEXT NOT NULL,
  action TEXT NOT NULL, answer TEXT NOT NULL, created_at REAL NOT NULL, response_json TEXT,
  PRIMARY KEY (chat_id, seq)
);
"""


class ChatStore:
    def __init__(self, path: Path | str | None):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._lock = threading.Lock()

    def create(self, sub: str, title: str = "New chat") -> dict[str, Any]:
        cid = "chat_" + uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute("INSERT INTO chats VALUES (?,?,?,?,?)",
                               (cid, sub, title[:120], now, now))  # fmt: skip
        return self.get(cid)  # type: ignore[return-value]

    def get(self, chat_id: str) -> dict[str, Any] | None:
        r = self._conn.execute("SELECT * FROM chats WHERE chat_id=?", (chat_id,)).fetchone()
        return dict(r) if r else None

    def owned(self, chat_id: str, sub: str) -> dict[str, Any] | None:
        c = self.get(chat_id)
        return c if c and c["sub"] == sub else None

    def list_for(self, sub: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM chats WHERE sub=? ORDER BY updated_at DESC LIMIT ?", (sub, limit))
        return [dict(r) for r in rows]

    def messages(self, chat_id: str, *, with_response: bool = False) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM messages WHERE chat_id=? ORDER BY seq",
                                  (chat_id,)).fetchall()  # fmt: skip
        out = []
        for r in rows:
            d = dict(r)
            rj = d.pop("response_json")
            if with_response:
                d["response"] = json.loads(rj) if rj else None
            out.append(d)
        return out

    def history(self, chat_id: str) -> list[str]:
        return [m["question"] for m in self.messages(chat_id)]

    def append(self, chat_id: str, *, request_id: str, question: str, action: str, answer: str,
               response: dict | None) -> int:  # fmt: skip
        with self._lock, self._conn:
            seq = self._conn.execute("SELECT COALESCE(MAX(seq),0)+1 FROM messages WHERE chat_id=?",
                                     (chat_id,)).fetchone()[0]  # fmt: skip
            self._conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)",
                               (chat_id, seq, request_id, question, action, answer, time.time(),
                                json.dumps(response, default=str) if response else None))
            title_sql = "UPDATE chats SET updated_at=?, title=CASE WHEN title='New chat' THEN ? " \
                        "ELSE title END WHERE chat_id=?"
            self._conn.execute(title_sql, (time.time(), question[:80], chat_id))
        return seq

    def delete(self, chat_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
            self._conn.execute("DELETE FROM chats WHERE chat_id=?", (chat_id,))
