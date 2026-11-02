"""Trial-request leads from the landing page: validated, appended to JSONL, optionally forwarded
to a webhook. No e-mail is sent (no SMTP dependency); the owner reads the list (admin endpoint)
and issues prospect tokens with ``scripts/issue_client_token.py``. A honeypot field catches bots:
such submissions are answered 202 but never stored."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

log = logging.getLogger("litbiorag.leads")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class LeadIn(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    email: str = Field(..., min_length=5, max_length=200)
    institution: str = Field("", max_length=200)
    role: str = Field("faculty", pattern="^(faculty|it|student|other)$")
    path: str = Field("institution", pattern="^(institution|student)$")
    course_size: str = Field("", max_length=40)
    message: str = Field("", max_length=2000)
    website: str = Field("", max_length=200)  # honeypot: humans never fill it

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip()
        if not _EMAIL.match(v):
            raise ValueError("enter a valid e-mail address")
        return v


class LeadStore:
    def __init__(self, path: Path | str | None, webhook_env: str = "LEADS_WEBHOOK_URL"):
        self.path = Path(path) if path else None
        self.webhook_env = webhook_env
        self._lock = threading.Lock()

    def add(self, lead: LeadIn, *, request_id: str, client: str = "") -> dict[str, Any]:
        rec = {"lead_id": "lead_" + uuid.uuid4().hex[:10], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "request_id": request_id, "client": client, "status": "new",
               **lead.model_dump(exclude={"website"})}  # fmt: skip
        if self.path:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
        self._webhook(rec)
        return rec

    def _webhook(self, rec: dict[str, Any]) -> None:
        url = os.environ.get(self.webhook_env, "")
        if not url:
            return
        try:
            req = urllib.request.Request(url, json.dumps(rec).encode(),
                                         headers={"Content-Type": "application/json"})  # fmt: skip
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception as e:  # noqa: BLE001  (a webhook failure never loses the lead)
            log.warning("lead webhook failed lead_id=%s: %s", rec["lead_id"], type(e).__name__)

    def list(self, limit: int = 200) -> list[dict[str, Any]]:
        if not self.path or not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out[-limit:][::-1]
