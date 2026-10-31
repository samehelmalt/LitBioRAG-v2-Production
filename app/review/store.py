"""SQLite store for human review cases (blocked or escalated answers awaiting a decision)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

STATUSES = ("pending", "released", "blocked", "expired")
_DDL = """
CREATE TABLE IF NOT EXISTS review_cases (
  case_id TEXT PRIMARY KEY, request_id TEXT NOT NULL, created_at REAL NOT NULL,
  sub TEXT NOT NULL, question TEXT NOT NULL, course_ids TEXT, action TEXT NOT NULL,
  internal_state TEXT NOT NULL, rule TEXT NOT NULL, reason TEXT NOT NULL,
  response_json TEXT NOT NULL, evidence_json TEXT NOT NULL, status TEXT NOT NULL,
  reviewer TEXT, decided_at REAL, reviewer_notes TEXT, released_answer TEXT,
  released_claim_ids TEXT, reverification_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_cases_status ON review_cases (status, created_at);
CREATE INDEX IF NOT EXISTS ix_cases_sub ON review_cases (sub, created_at);
"""


class ReviewStore:
    def __init__(self, path: Path | str | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path) if self.path else ":memory:",
                                     check_same_thread=False)  # fmt: skip
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._lock = threading.Lock()

    def create(self, *, request_id: str, sub: str, question: str, course_ids: list[str] | None,
               action: str, internal_state: str, rule: str, reason: str, response: dict,
               evidence: list[dict]) -> dict:  # fmt: skip
        case_id = uuid.uuid4().hex[:20]
        with self._lock:
            self._conn.execute(
                "INSERT INTO review_cases VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (case_id, request_id, time.time(), sub, question, json.dumps(course_ids or []),
                 action, internal_state, rule, reason, json.dumps(response, default=str),
                 json.dumps(evidence, default=str), "pending", None, None, None, None, None, None),
            )
            self._conn.commit()
        return self.get(case_id)  # type: ignore[return-value]

    def get(self, case_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM review_cases WHERE case_id = ?",
                                 (case_id,)).fetchone()  # fmt: skip
        return _row(row) if row else None

    def list(self, *, status: str | None = "pending", sub: str | None = None,
             limit: int = 50) -> list[dict]:  # fmt: skip
        q, args = "SELECT * FROM review_cases WHERE 1=1", []
        if status:
            q += " AND status = ?"
            args.append(status)
        if sub:
            q += " AND sub = ?"
            args.append(sub)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        return [_row(r) for r in self._conn.execute(q, args).fetchall()]

    def decide(self, case_id: str, *, status: str, reviewer: str, notes: str,
               released_answer: str | None, released_claim_ids: list[str] | None,
               reverification: dict | None) -> dict:  # fmt: skip
        assert status in ("released", "blocked")
        with self._lock:
            cur = self._conn.execute(
                "UPDATE review_cases SET status=?, reviewer=?, decided_at=?, reviewer_notes=?, "
                "released_answer=?, released_claim_ids=?, reverification_json=? "
                "WHERE case_id=? AND status='pending'",
                (status, reviewer, time.time(), notes, released_answer,
                 json.dumps(released_claim_ids or []), json.dumps(reverification, default=str),
                 case_id),
            )
            self._conn.commit()
        if cur.rowcount != 1:
            raise LookupError("case is not pending")
        return self.get(case_id)  # type: ignore[return-value]

    def expire(self, max_age_s: float, now: float | None = None) -> int:
        cutoff = (now or time.time()) - max_age_s
        with self._lock:
            cur = self._conn.execute(
                "UPDATE review_cases SET status='expired' WHERE status='pending' AND created_at<?",
                (cutoff,),
            )
            self._conn.commit()
        return cur.rowcount

    def counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) FROM review_cases GROUP BY status").fetchall()
        return {s: 0 for s in STATUSES} | {r[0]: r[1] for r in rows}


def _row(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    for k in ("course_ids", "response_json", "evidence_json", "released_claim_ids",
              "reverification_json"):
        if d.get(k):
            d[k] = json.loads(d[k])
    d["response"] = d.pop("response_json")
    d["evidence"] = d.pop("evidence_json")
    d["reverification"] = d.pop("reverification_json")
    return d
