"""Human-in-the-loop review: a blocked or escalated answer waits for a reviewer, who releases it
(optionally edited, re-verified against the same evidence) or keeps it blocked. Every decision
is audited and exported as a label record for calibration.

Why this is safe: the machine gate still decides first and still blocks; a human can only
*add* a release, and an edited release is re-verified by the same verifier so that unsupported
text cannot be released silently (the reviewer must confirm with the re-verification in view).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.api.audit import AuditLog
from app.evidence.selector import EvidenceUnit
from app.generation.contract import Claim, GeneratedAnswer
from app.retrieval.passages import split_sentences
from app.review.store import ReviewStore
from app.verification.pipeline import verify

_ROOT = Path(__file__).resolve().parents[2]


class ReviewConfig(BaseModel):
    enqueue: dict[str, list[str]] = Field(default_factory=lambda: {
        "actions": ["ABSTAIN", "ESCALATE"], "internal_states": ["BLOCK"],
        "skip_reasons": ["no_evidence", "resource_limit"]})  # fmt: skip
    expire_hours: float = 72.0
    db_path: str = "data/review/review.sqlite"
    labels_path: str = "data/audit/labels_from_review.jsonl"
    reverify_on_release: bool = True

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> ReviewConfig:
        p = Path(path or os.environ.get("LITBIORAG_REVIEW_CONFIG")
                 or _ROOT / "configs" / "review.yaml")  # fmt: skip
        return cls.model_validate(yaml.safe_load(p.read_text()) or {}) if p.exists() else cls()


class ReleaseNeedsConfirmation(Exception):
    def __init__(self, reverification: dict):
        super().__init__("re-verification found unsupported or contradicted claims")
        self.reverification = reverification


class ReviewService:
    def __init__(self, pipeline, cfg: ReviewConfig | None = None, *,
                 store: ReviewStore | None = None, labels_path: Path | None = None,
                 audit: AuditLog | None = None):  # fmt: skip
        self.pipeline = pipeline
        self.cfg = cfg or ReviewConfig.from_yaml()
        self.store = store or ReviewStore(_ROOT / self.cfg.db_path)
        self.labels = AuditLog(labels_path if labels_path is not None
                               else _ROOT / self.cfg.labels_path)  # fmt: skip
        self.audit = audit or AuditLog(None)
        self._lock = threading.Lock()

    # ---- enqueue -------------------------------------------------------------------------
    def should_enqueue(self, action: str, internal_state: str, abstain_reason: str | None) -> bool:
        e = self.cfg.enqueue
        return (action in e.get("actions", []) and internal_state in e.get("internal_states", [])
                and (abstain_reason or "") not in e.get("skip_reasons", []))

    def maybe_enqueue(self, out, response: dict, *, sub: str,
                      course_ids: list[str] | None) -> dict | None:  # fmt: skip
        d = out.decision
        if not self.should_enqueue(d.action.value, d.internal_state, out.generated.abstain_reason):
            return None
        case = self.store.create(
            request_id=response["request_id"], sub=sub, question=out.question,
            course_ids=course_ids, action=d.action.value, internal_state=d.internal_state,
            rule=d.rule, reason=d.reason, response=response,
            evidence=[u.model_dump(mode="json") for u in out.evidence],
        )
        self.audit.record({"event": "review_enqueued", "case_id": case["case_id"],
                           "request_id": case["request_id"], "rule": d.rule})  # fmt: skip
        return case

    # ---- decide --------------------------------------------------------------------------
    def reverify(self, case: dict, text: str) -> dict:
        """Re-run claim verification on reviewer text against the case's evidence."""
        units = [EvidenceUnit.model_validate(u) for u in case["evidence"]]
        spans = split_sentences(text)
        claims = [Claim(id=f"R{i + 1}", text=text[s:e].strip(), citations=[], critical=False)
                  for i, (s, e) in enumerate(spans) if len(text[s:e].strip()) >= 8]  # fmt: skip
        ans = GeneratedAnswer(answer=text, claims=claims, question=case["question"])
        rep = verify(ans, units, self.pipeline.pool.acquire("verifier"),
                     self.pipeline.pool.acquire("entity_extractor"), config=self.pipeline.ver_cfg,
                     thresholds=self.pipeline.registry.thresholds("verifier"),
                     question=case["question"])  # fmt: skip
        flagged = [v.claim_id for v in rep.verdicts if v.label != "supported"]
        return {
            "n_claims": len(rep.verdicts), "entailment_rate": rep.entailment_rate,
            "contradiction_rate": rep.contradiction_rate, "unsupported_rate": rep.unsupported_rate,
            "substitution_detected": rep.substitution_detected, "flagged_claims": flagged,
            "clean": not flagged and not rep.substitution_detected,
            "verdicts": [v.model_dump(mode="json") for v in rep.verdicts],
        }

    def decide(self, case_id: str, *, decision: str, reviewer: str, notes: str = "",
               edited_answer: str | None = None, kept_claim_ids: list[str] | None = None,
               confirm: bool = False) -> dict:  # fmt: skip
        case = self.store.get(case_id)
        if case is None:
            raise LookupError("unknown case")
        if case["status"] != "pending":
            raise LookupError("case is not pending")
        reverification = None
        released_answer = None
        if decision == "release":
            released_answer = self._release_text(case, edited_answer, kept_claim_ids)
            if self.cfg.reverify_on_release:
                reverification = self.reverify(case, released_answer)
                if not reverification["clean"] and not confirm:
                    raise ReleaseNeedsConfirmation(reverification)
            status = "released"
        elif decision == "block":
            status = "blocked"
        else:
            raise ValueError("decision must be release or block")
        updated = self.store.decide(case_id, status=status, reviewer=reviewer, notes=notes,
                                    released_answer=released_answer,
                                    released_claim_ids=kept_claim_ids,
                                    reverification=reverification)  # fmt: skip
        self.audit.record({"event": "review_decision", "case_id": case_id, "decision": status,
                           "reviewer": reviewer, "confirmed_despite_flags": bool(
                               reverification and not reverification["clean"])})  # fmt: skip
        self._export_label(updated)
        return updated

    def _release_text(self, case: dict, edited: str | None, kept: list[str] | None) -> str:
        if edited and edited.strip():
            return edited.strip()
        claims = case["response"].get("claims", [])
        if kept:
            texts = [c["text"] for c in claims if c["id"] in set(kept)]
        else:
            texts = [c["text"] for c in claims if not c.get("removed")]
        if not texts:
            raise ValueError("nothing to release: provide edited_answer or kept_claim_ids")
        return " ".join(texts)

    def _export_label(self, case: dict) -> None:
        """LabelRecord-shaped dict (evaluation/labels.py); app/ never imports evaluation/."""
        rec: dict[str, Any] = {
            "query_id": case["request_id"], "question": case["question"],
            "answer": case["released_answer"] or case["response"].get("answer", ""),
            "evidence": [{"id": u.get("passage_id"), "title": u.get("title"), "year": u.get("year"),
                          "source_type": u.get("source_type"), "text": u.get("text")}
                         for u in case["evidence"]],
            "label": "acceptable" if case["status"] == "released" else "unacceptable",
            "rater": case["reviewer"] or "", "notes": case["reviewer_notes"] or "",
            "run_id": json.dumps({k: case["response"].get("versions", {}).get(k)
                                  for k in ("profile", "bundle_id", "prompt_id")}),
            "proxy_label": None,
            "machine_action": case["action"], "machine_rule": case["rule"],
            "edited": bool(case["released_answer"]) and case["released_answer"]
            != case["response"].get("answer"),
            "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(case["decided_at"])),
        }  # fmt: skip
        self.labels.record(rec)

    def expire(self) -> int:
        return self.store.expire(self.cfg.expire_hours * 3600)
