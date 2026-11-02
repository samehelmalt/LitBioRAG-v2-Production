"""Trial allowances: a fixed number of answered questions per authenticated identity.

Used for client prospects: the allowance is reserved before the pipeline runs (so a crash cannot
grant a free question), refunded when the request fails before an answer is produced, charged at
most once per request id (the POST and the SSE path share the id), persisted in SQLite so a
restart does not reset it, and never available to anonymous callers. Admins are exempt. This is
a trial counter, not a billing or anti-fraud system; identity is only as strong as the token issuer.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path


class TrialLimitExceeded(Exception):
    pass


class TrialLedger:
    def __init__(self, path: Path | str | None, limit: int = 5):
        if limit < 1:
            raise ValueError("limit must be positive")
        self.limit = limit
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False,
                                     timeout=30)  # fmt: skip
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS trial_uses (request_id TEXT PRIMARY KEY, "
            "subject TEXT NOT NULL, used_at REAL NOT NULL)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS ix_trial_subject ON trial_uses (subject)")
        self._conn.commit()
        self._lock = threading.RLock()

    def used(self, subject: str) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM trial_uses WHERE subject=?",
                                      (subject,)).fetchone()[0]  # fmt: skip

    def remaining(self, subject: str) -> int:
        return max(0, self.limit - self.used(subject))

    def reserve(self, subject: str, request_id: str) -> int:
        """Charge one question; idempotent per request id. Returns the remaining allowance."""
        if not subject or subject == "anonymous":
            raise TrialLimitExceeded("sign in to use the trial")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT subject FROM trial_uses WHERE request_id=?",
                                         (request_id,)).fetchone()  # fmt: skip
                if row is not None:
                    if row[0] != subject:
                        raise TrialLimitExceeded("request id belongs to another identity")
                else:
                    n = self._conn.execute("SELECT COUNT(*) FROM trial_uses WHERE subject=?",
                                           (subject,)).fetchone()[0]  # fmt: skip
                    if n >= self.limit:
                        raise TrialLimitExceeded(
                            f"trial allowance of {self.limit} questions is used up")
                    self._conn.execute("INSERT INTO trial_uses VALUES (?,?,?)",
                                       (request_id, subject, time.time()))  # fmt: skip
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return self.remaining(subject)

    def refund(self, subject: str, request_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM trial_uses WHERE subject=? AND request_id=?",
                                     (subject, request_id))  # fmt: skip
            return cur.rowcount > 0
