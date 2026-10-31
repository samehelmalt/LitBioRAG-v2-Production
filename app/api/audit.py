"""Audit trail: one JSON line per request, plus a structured log line (observability §31)."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("litbiorag.api")


class AuditLog:
    def __init__(self, path: Path | None, *, store_question_text: bool = True):
        self.path = path
        self.store_question_text = store_question_text
        self._lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: dict[str, Any]) -> None:
        event = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
        if not self.store_question_text and "question" in event:
            q = event.pop("question")
            event["question_sha256"] = hashlib.sha256(q.encode("utf-8")).hexdigest()
        line = json.dumps(event, ensure_ascii=False, default=str)
        log.info(line)
        if self.path:
            with self._lock, open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
