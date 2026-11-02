"""One request → one immutable answer record with six channels.

RAW        every generation attempt's exact output (and hidden reasoning when a profile strips it)
CANDIDATE  parsed claims per attempt, before any release decision
USER       what the asker sees: the delivered answer or the abstention text, with citations
AUDIT      evidence, claim verdicts, quality reports, attempts, gate trace, timings, usage
CONTROL    versions, profile, scope, budget/operating point, request metadata
HISTORY    what a conversation may carry forward: only delivered answers (claims + citations +
           entities); withheld answers contribute nothing

The API response, the chat history, the review queue and the audit view are all rendered from
this record, so they can never disagree about what was answered for a request id. Records are
append-only (SQLite); nothing is updated after insertion.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

CHANNELS = ("raw", "candidate", "user", "audit", "control", "history")
DELIVERED_ACTIONS = ("ANSWER", "ANSWER_WITH_CAUTION")

_DDL = """
CREATE TABLE IF NOT EXISTS answer_records (
  request_id TEXT PRIMARY KEY, created_at REAL NOT NULL, sub TEXT NOT NULL, chat_id TEXT,
  action TEXT NOT NULL, delivered INTEGER NOT NULL, question TEXT NOT NULL,
  raw TEXT NOT NULL, candidate TEXT NOT NULL, user TEXT NOT NULL, audit TEXT NOT NULL,
  control TEXT NOT NULL, history TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_records_sub ON answer_records (sub, created_at);
CREATE INDEX IF NOT EXISTS ix_records_chat ON answer_records (chat_id, created_at);
"""


def build_record(out, response: dict[str, Any], *, sub: str, chat_id: str | None,
                 scope: str, kb_ids: list[str], versions: dict[str, Any],
                 operating_point: dict | None) -> dict[str, Any]:  # fmt: skip
    """Assemble the channels from a PipelineOutput and the rendered QueryResponse dict."""
    delivered = response["action"] in DELIVERED_ACTIONS
    attempts = list(getattr(out, "attempts", []) or [])
    raw = [{"n": a.n, "raw": a.raw, "selected": a.selected, "feedback_given": a.feedback_given,
            "reasoning": getattr(a, "reasoning", "")} for a in attempts]  # fmt: skip
    candidate = [{"n": a.n, "abstained": a.answer.abstained,
                  "abstain_reason": a.answer.abstain_reason,
                  "claims": [c.model_dump(mode="json") for c in a.answer.claims],
                  "conflicts": [c.model_dump(mode="json") for c in a.answer.conflicts],
                  "quality": a.quality.to_dict(), "vetoed": a.vetoed,
                  "veto_reason": a.veto_reason, "selected": a.selected}
                 for a in attempts]  # fmt: skip
    user = {
        "action": response["action"], "answer": response["answer"],
        "abstained": response["abstained"], "abstain_reason": response["abstain_reason"],
        "citations": sorted({c for cl in response["claims"] if not cl["removed"]
                             for c in cl["citations"]}),
        "sources": [{k: s[k] for k in ("citation_id", "display_source", "tier", "url")}
                    for s in response["sources"]],
        "review": response.get("review"), "disclosure": response["disclosure"],
    }  # fmt: skip
    audit = {
        "rule": response["rule"], "reason": response["reason"], "trace": response["trace"],
        "confidence": response["confidence"],
        "confidence_calibrated": response["confidence_calibrated"],
        "sufficiency": response["sufficiency"], "claims": response["claims"],
        "removed_claims": response["removed_claims"], "conflicts": response["conflicts"],
        "entities": response["entities"], "verification": response["verification"],
        "attempts": [a.summary() for a in attempts],
        "evidence": [{k: s[k] for k in ("citation_id", "passage_id", "doc_id", "char_start",
                                        "char_end", "retrieval_score", "rank", "tier")}
                     for s in response["sources"]],
        "evidence_fit": response.get("evidence_fit"), "usage": response.get("usage"),
        "timings_ms": response["timings_ms"],
    }  # fmt: skip
    control = {
        "versions": versions, "scope": scope, "kb_ids": kb_ids,
        "operating_point": operating_point, "request_id": response["request_id"],
        "interpreted_as": response.get("interpreted_as"),
        "sub": sub, "chat_id": chat_id, "created_at": time.time(),
    }  # fmt: skip
    history = (
        {"question": out.question, "answer": response["answer"],
         "claims": [{"id": cl["id"], "text": cl["text"], "citations": cl["citations"]}
                    for cl in response["claims"] if not cl["removed"]],
         "entities": response["entities"]["asked"], "citations": user["citations"],
         "action": response["action"]}
        if delivered else {"question": out.question, "withheld": True,
                           "action": response["action"], "review": response.get("review")}
    )  # fmt: skip
    return {"request_id": response["request_id"], "sub": sub, "chat_id": chat_id,
            "action": response["action"], "delivered": delivered, "question": out.question,
            "raw": raw, "candidate": candidate, "user": user, "audit": audit,
            "control": control, "history": history}  # fmt: skip


class RecordStore:
    def __init__(self, path: Path | str | None):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._lock = threading.Lock()

    def save(self, rec: dict[str, Any]) -> None:
        """Insert once; a second save for the same request id is refused (append-only)."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO answer_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec["request_id"], time.time(), rec["sub"], rec["chat_id"], rec["action"],
                 int(rec["delivered"]), rec["question"],
                 *(json.dumps(rec[c], default=str) for c in CHANNELS)),
            )

    def get(self, request_id: str) -> dict[str, Any] | None:
        r = self._conn.execute("SELECT * FROM answer_records WHERE request_id=?",
                               (request_id,)).fetchone()  # fmt: skip
        if r is None:
            return None
        d = dict(r)
        for c in CHANNELS:
            d[c] = json.loads(d[c])
        d["delivered"] = bool(d["delivered"])
        return d

    def history_for_chat(self, chat_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT history FROM answer_records WHERE chat_id=? ORDER BY created_at", (chat_id,))
        return [json.loads(r[0]) for r in rows]

    def list_for(self, sub: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT request_id, created_at, chat_id, action, delivered, question FROM "
            "answer_records WHERE sub=? ORDER BY created_at DESC LIMIT ?", (sub, limit))
        return [dict(r) for r in rows]


def history_as_context(history: list[dict[str, Any]], *, max_turns: int = 4,
                       max_chars: int = 300) -> list[str]:  # fmt: skip
    """Render delivered turns as context strings for the generator (never as evidence)."""
    out: list[str] = []
    for h in history:
        if h.get("withheld"):
            continue
        ans = (h.get("answer") or "")[:max_chars]
        out.append(f"Q: {h['question'][:max_chars]} → delivered answer: {ans}")
    return out[-max_turns:]
