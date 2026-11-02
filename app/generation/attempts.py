"""Bounded, evidence-guided regeneration with an immutable attempt ledger.

One request is one transaction that may contain several generation attempts. Each attempt is
recorded as produced (raw text, parsed claims, quality report, verification summary); nothing
is edited. The first attempt that passes the quality checks and has no verification veto is
selected; otherwise the best verified attempt is selected only if it has no quality defects;
otherwise the transaction abstains with reason ``generation_quality``. Every user-facing and
diagnostic channel is rendered from the selected attempt (``app/service/records.py``), so the
chat, the audit view and the review queue can never show different answers for one request.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core.verdicts import VerificationReport
from app.evidence.selector import EvidenceUnit
from app.generation.contract import ABSTENTION_TEXT, GeneratedAnswer
from app.generation.quality import QualityReport, assess, feedback_for

ABSTAIN_GENERATION_QUALITY = "generation_quality"
ABSTENTION_TEXT.setdefault(
    ABSTAIN_GENERATION_QUALITY,
    "The system could not produce an answer that passed its output-quality and verification "
    "checks for this question.",
)


@dataclass
class AttemptRecord:
    n: int
    raw: str
    answer: GeneratedAnswer
    quality: QualityReport
    verification: VerificationReport | None = None
    vetoed: bool = False
    veto_reason: str = ""
    selected: bool = False
    feedback_given: str = ""
    started_at: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    def summary(self) -> dict[str, Any]:
        v = self.verification
        return {
            "n": self.n, "selected": self.selected, "quality_ok": self.quality.ok,
            "defects": list(self.quality.defects), "vetoed": self.vetoed,
            "veto_reason": self.veto_reason, "abstained": self.answer.abstained,
            "abstain_reason": self.answer.abstain_reason, "n_claims": len(self.answer.claims),
            "entailment_rate": v.entailment_rate if v else None,
            "unsupported_rate": v.unsupported_rate if v else None,
            "fabricated_citations": list(v.fabricated_citations) if v else [],
            "substitution_detected": v.substitution_detected if v else None,
            "usage": self.answer.usage, "duration_ms": round(self.duration_ms, 1),
            "feedback_given": self.feedback_given,
        }  # fmt: skip


def hard_veto(report: VerificationReport) -> str:
    """Verification outcomes that make an attempt unusable regardless of its score."""
    if report.fabricated_citations:
        return f"fabricated citation(s): {', '.join(report.fabricated_citations)}"
    if report.substitution_detected:
        return "entity substitution"
    crit = [v for v in report.verdicts if v.critical]
    if any(v.label == "contradicted" for v in crit):
        return "critical claim contradicted by the evidence"
    if any(v.numeric_consistent is False for v in crit):
        return "critical numeric mismatch"
    return ""


def verification_notes(report: VerificationReport) -> list[str]:
    notes: list[str] = []
    for v in report.verdicts:
        if v.fabricated_ids:
            notes.append(f"claim {v.claim_id} cited {', '.join(v.fabricated_ids)}, which is not in "
                         "the evidence")  # fmt: skip
        elif v.label == "contradicted":
            notes.append(f"claim {v.claim_id} is contradicted by the cited evidence")
        elif v.label == "unsupported" and v.critical:
            notes.append(f"claim {v.claim_id} is not supported by the passages it cites")
        if v.near_neighbour_substitution and v.substituted_for:
            ce, _, qe = v.substituted_for.partition("->")
            notes.append(f"claim {v.claim_id} is about {ce}; the question asked about {qe}")
        if v.numeric_mismatches:
            notes.append(f"claim {v.claim_id} states numbers not in the evidence: "
                         + ", ".join(v.numeric_mismatches))  # fmt: skip
    return notes


def run_attempts(
    generate: Callable[[str | None], tuple[str, GeneratedAnswer]],
    verify: Callable[[GeneratedAnswer], VerificationReport],
    *,
    max_attempts: int = 2,
    question_entities: set[str] | None = None,
    extract_entities: Callable[[str], set[str]] | None = None,
    units: Sequence[EvidenceUnit] = (),
) -> tuple[list[AttemptRecord], AttemptRecord]:
    """``generate(feedback)`` returns (raw, parsed); ``verify(parsed)`` returns the report.
    Returns (all attempts, selected attempt); the selected attempt may be an abstention."""
    attempts: list[AttemptRecord] = []
    feedback: str | None = None
    for n in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        raw, parsed = generate(feedback)
        claim_ents: set[str] | None = None
        if extract_entities is not None and parsed.claims:
            claim_ents = set()
            for c in parsed.claims:
                claim_ents |= extract_entities(c.text)
        q = assess(raw, parsed, question_entities=question_entities, claim_entities=claim_ents)
        rec = AttemptRecord(n=n, raw=raw, answer=parsed, quality=q,
                            feedback_given=feedback or "")  # fmt: skip
        if not parsed.abstained:
            rec.verification = verify(parsed)
            rec.veto_reason = hard_veto(rec.verification)
            rec.vetoed = bool(rec.veto_reason)
        rec.duration_ms = (time.perf_counter() - t0) * 1000
        attempts.append(rec)
        if parsed.abstained and parsed.format_ok:
            break  # a deliberate abstention is final; no regeneration to force an answer
        if q.ok and not rec.vetoed:
            break
        feedback = feedback_for(q, verification_notes(rec.verification) if rec.verification
                                else None)  # fmt: skip
    selected = _select(attempts)
    selected.selected = True
    return attempts, selected


def _select(attempts: list[AttemptRecord]) -> AttemptRecord:
    clean = [a for a in attempts if a.quality.ok and not a.vetoed and not a.answer.abstained]
    if clean:
        return max(clean, key=lambda a: (a.verification.entailment_rate or 0.0)
                   if a.verification else 0.0)  # fmt: skip
    deliberate = [a for a in attempts if a.answer.abstained and a.answer.format_ok]
    if deliberate:
        return deliberate[-1]
    # every attempt was defective or vetoed: abstain, keep the attempts for audit
    last = attempts[-1]
    reasons = "; ".join(last.quality.defects + ([last.veto_reason] if last.veto_reason else []))
    abst = GeneratedAnswer.abstention(ABSTAIN_GENERATION_QUALITY, raw=last.raw,
                                      prompt_id=last.answer.prompt_id,
                                      model=last.answer.model)  # fmt: skip
    abst.question = last.answer.question
    abst.usage = last.answer.usage
    abst.evidence_fit = last.answer.evidence_fit
    rec = AttemptRecord(n=last.n, raw=last.raw, answer=abst, quality=last.quality,
                        verification=last.verification, vetoed=last.vetoed,
                        veto_reason=reasons)  # fmt: skip
    attempts.append(rec)
    return rec
