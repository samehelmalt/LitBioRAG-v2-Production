"""Gate v2: ordered vetoes, then one calibrated decision. No composite score.

Rules are applied in a fixed order; the first rule that fires decides, and its id is recorded in
the trace. Every rule reads only runtime fields of ``VerificationReport`` / ``GeneratedAnswer``;
gold labels are never available here (a test greps ``app/`` for them).

    1  scope             patient-specific / personal dosing / emergency   → ESCALATE
    2  generator abstained                                                → ABSTAIN
    3  fabricated citation anywhere                                       → ABSTAIN
    4  critical claim uncited or its citation does not support it         → ABSTAIN
    5  critical claim contradicted                                        → ABSTAIN (conflict text)
    6  entity substitution / critical entity mismatch                     → ABSTAIN
    7  critical numeric mismatch                                          → ABSTAIN
    8  question entities absent from the evidence                         → CLARIFY
    9  evidence sufficiency below threshold                               → ABSTAIN
    10 non-critical unsupported and off-target claims removed (repair)    → ANSWER_WITH_CAUTION
    11 conflict recorded on any claim                                     → ANSWER_WITH_CAUTION
    12 calibrated confidence ≥ τ                                          → ANSWER else CAUTION

Internal state PASS/WARN/BLOCK is kept for comparability with the thesis rows.
"""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.core.verdicts import VerificationReport
from app.generation.contract import ABSTENTION_TEXT, GeneratedAnswer
from app.safety.repair import repair_answer
from app.safety.scope import ScopeDecision, classify_scope


class Action(str, Enum):
    ANSWER = "ANSWER"
    ANSWER_WITH_CAUTION = "ANSWER_WITH_CAUTION"
    CLARIFY = "CLARIFY"
    ABSTAIN = "ABSTAIN"
    ESCALATE = "ESCALATE"


INTERNAL_STATE = {
    Action.ANSWER: "PASS",
    Action.ANSWER_WITH_CAUTION: "WARN",
    Action.CLARIFY: "BLOCK",
    Action.ABSTAIN: "BLOCK",
    Action.ESCALATE: "BLOCK",
}


class GateConfig(BaseModel):
    unsupported_noncritical_max: float = Field(0.34, ge=0, le=1)  # share of claims removable
    grounding_min: float = Field(0.5, ge=0, le=1)  # question entities present in evidence
    sufficiency_min: float = Field(0.35, ge=0, le=1)
    confidence_answer: float = Field(0.70, ge=0, le=1)  # ANSWER vs ANSWER_WITH_CAUTION
    confidence_model: dict | None = None  # {"weights": {...}, "bias": b}; fitted by calibrate_gate
    sufficiency_model: dict | None = None
    costs: dict[str, float] = Field(
        default_factory=lambda: {
            "false_pass": 10.0, "false_block": 1.0, "warn": 0.3, "abstain": 1.0
        }  # fmt: skip
    )
    calibrated_on: str = ""  # run id of the calibration job; empty = heuristic defaults
    operating_point: dict | None = None  # written by calibrate_gate: mode, target, precision…

    @classmethod
    def from_yaml(cls, path: Path | str) -> GateConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})


class Decision(BaseModel):
    action: Action
    internal_state: str
    rule: str  # id of the deciding rule
    reason: str
    confidence: float | None = None
    sufficiency: float | None = None
    answer_text: str
    removed_claims: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)


def confidence_features(report: VerificationReport, sufficiency: float | None) -> dict[str, float]:
    return {
        "entailment_rate": report.entailment_rate or 0.0,
        "citation_support_rate": (
            report.citation_support_rate if report.citation_support_rate is not None else 0.0
        ),
        "not_unsupported": 1.0 - (report.unsupported_rate or 0.0),
        "entity_fidelity": report.entity_fidelity if report.entity_fidelity is not None else 1.0,
        "numeric_fidelity": report.numeric_fidelity if report.numeric_fidelity is not None else 1.0,
        "query_entity_grounding": (
            report.query_entity_grounding if report.query_entity_grounding is not None else 1.0
        ),
        "sufficiency": sufficiency if sufficiency is not None else 0.5,
    }


def confidence(report: VerificationReport, sufficiency: float | None, model: dict | None) -> float:
    f = confidence_features(report, sufficiency)
    if model and model.get("weights"):
        z = float(model.get("bias", 0.0)) + sum(
            float(w) * f.get(k, 0.0) for k, w in model["weights"].items()
        )
        return 1.0 / (1.0 + math.exp(-z))
    return min(f.values())  # uncalibrated: the weakest check bounds confidence


def decide(
    report: VerificationReport,
    answer: GeneratedAnswer,
    *,
    sufficiency: float | None = None,
    config: GateConfig | None = None,
    scope: ScopeDecision | None = None,
    question: str | None = None,
) -> Decision:
    cfg = config or GateConfig()
    trace: list[str] = []
    q = question if question is not None else answer.question
    scope = scope or classify_scope(q)

    def out(action: Action, rule: str, reason: str, text: str, removed=None, conf=None) -> Decision:
        trace.append(f"{rule}: {action.value} — {reason}")
        return Decision(
            action=action, internal_state=INTERNAL_STATE[action], rule=rule, reason=reason,
            confidence=conf, sufficiency=sufficiency, answer_text=text,
            removed_claims=removed or [], trace=trace,
        )  # fmt: skip

    # 1 scope
    if not scope.in_scope:
        return out(Action.ESCALATE, "R1-scope", scope.kind, scope.message)
    trace.append("R1-scope: in scope")
    # 2 generator abstained
    if answer.abstained:
        reason = answer.abstain_reason or "insufficient_evidence"
        return out(Action.ABSTAIN, "R2-generator-abstained", reason, answer.answer)
    trace.append("R2-generator-abstained: no")
    crit = [v for v in report.verdicts if v.critical]
    # 3 fabricated citations
    if report.fabricated_citations:
        return out(Action.ABSTAIN, "R3-fabricated-citation",
                   f"cited ids not in evidence: {', '.join(report.fabricated_citations)}",
                   ABSTENTION_TEXT["insufficient_evidence"])  # fmt: skip
    trace.append("R3-fabricated-citation: none")
    # 4 critical claims must be cited and the citation must support them
    bad = [v.claim_id for v in crit if not v.cited_ids or not v.supporting_ids]
    if bad:
        return out(Action.ABSTAIN, "R4-critical-uncited",
                   f"critical claim(s) without a supporting citation: {', '.join(bad)}",
                   ABSTENTION_TEXT["insufficient_evidence"])  # fmt: skip
    trace.append("R4-critical-uncited: none")
    # 5 critical contradiction
    bad = [v.claim_id for v in crit if v.label == "contradicted"]
    if bad:
        conflict = any(k.claim_id in bad for k in report.conflicts)
        head = "conflicting evidence on " if conflict else "evidence contradicts "
        reason = head + ", ".join(bad)
        text = ABSTENTION_TEXT["conflicting_evidence" if conflict else "insufficient_evidence"]
        return out(Action.ABSTAIN, "R5-critical-contradicted", reason, text)
    trace.append("R5-critical-contradicted: none")
    # 6 entity substitution / critical entity mismatch
    subs = [v.claim_id for v in report.verdicts if v.near_neighbour_substitution]
    mism = [v.claim_id for v in crit if v.entity_consistent is False and v.claim_id not in subs]
    bad = subs + mism
    if bad:
        return out(Action.ABSTAIN, "R6-entity",
                   ("entity substitution on " if subs else "entity mismatch on ") + ", ".join(bad),
                   ABSTENTION_TEXT["insufficient_evidence"])  # fmt: skip
    trace.append("R6-entity: consistent")
    # 6b off-topic: the evidence covers the asked entities but no critical claim names any of
    # them — the answer is about something else (trifluridine for an NSCLC question). A wrong-
    # topic answer is a false PASS, so this is a veto, not a caution.
    asked = set(report.question_entities)
    if crit and asked and set(report.question_entities_missing_in_claims) >= asked \
            and not set(report.question_entities_missing_in_evidence) >= asked:
        return out(Action.ABSTAIN, "R6-offtopic",
                   "no critical claim addresses the asked entities: " + ", ".join(sorted(asked)),
                   ABSTENTION_TEXT["insufficient_evidence"])  # fmt: skip
    trace.append("R6-offtopic: addressed")
    # 7 critical numeric mismatch
    bad = [v.claim_id for v in crit if v.numeric_consistent is False]
    if bad:
        return out(Action.ABSTAIN, "R7-numeric", f"numeric mismatch on {', '.join(bad)}",
                   ABSTENTION_TEXT["insufficient_evidence"])  # fmt: skip
    trace.append("R7-numeric: consistent")
    # 8 question entities absent from evidence → ask the user to clarify / reformulate
    g = report.query_entity_grounding
    if g is not None and g < cfg.grounding_min:
        missing = ", ".join(report.question_entities_missing_in_evidence) or "question entities"
        return out(Action.CLARIFY, "R8-grounding",
                   f"the retrieved evidence does not cover: {missing}",
                   "The retrieved literature does not cover all entities in the question. "
                   "Please check the terms or narrow the question.")  # fmt: skip
    trace.append(f"R8-grounding: {g if g is not None else 'n/a'}")
    # 9 sufficiency
    if sufficiency is not None and sufficiency < cfg.sufficiency_min:
        return out(Action.ABSTAIN, "R9-sufficiency", f"evidence sufficiency {sufficiency:.2f}",
                   ABSTENTION_TEXT["insufficient_evidence"])  # fmt: skip
    trace.append(f"R9-sufficiency: {sufficiency if sufficiency is not None else 'n/a'}")
    # 10 repair: remove non-critical unsupported claims and off-target claims (a near neighbour
    # named where the answer does cover the asked entity elsewhere)
    removable = [
        v.claim_id
        for v in report.verdicts
        if (not v.critical and v.label != "supported") or v.off_target
    ]
    n = max(report.n_claims, 1)
    if len(removable) / n > cfg.unsupported_noncritical_max:
        return out(Action.ABSTAIN, "R10-too-many-unsupported",
                   f"{len(removable)}/{n} claims unsupported",
                   ABSTENTION_TEXT["insufficient_evidence"])  # fmt: skip
    text, removed = repair_answer(answer, removable) if removable else (answer.answer, [])
    if removable:
        trace.append(f"R10-repair: removed {len(removable)} unsupported non-critical claim(s)")
    conf = confidence(report, sufficiency, cfg.confidence_model)
    # 11 conflicts → caution
    if report.conflicts:
        lec = [k.claim_id for k in report.conflicts if k.kind == "lecture_vs_literature"]
        other = [k.claim_id for k in report.conflicts if k.kind != "lecture_vs_literature"]
        parts = []
        if lec:
            parts.append("the course material and the literature differ on " + ", ".join(lec))
        if other:
            parts.append("mixed evidence on " + ", ".join(other))
        return out(Action.ANSWER_WITH_CAUTION, "R11-conflict", "; ".join(parts), text, removed,
                   conf)  # fmt: skip
    if removed:
        return out(Action.ANSWER_WITH_CAUTION, "R10-repair", "unsupported claims removed",
                   text, removed, conf)  # fmt: skip
    # 12 calibrated confidence
    action = Action.ANSWER if conf >= cfg.confidence_answer else Action.ANSWER_WITH_CAUTION
    return out(action, "R12-confidence", f"confidence {conf:.2f}", text, [], conf)
