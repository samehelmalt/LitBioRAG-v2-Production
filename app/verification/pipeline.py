"""Claim verification pipeline.

    GeneratedAnswer + EvidenceUnit[]  →  VerificationReport

Per claim: resolve citations (fabricated ids fail) → score (passage, claim) pairs with the
verifier → label from max entailment / max contradiction → entity consistency and numeric
consistency against the *supporting* passages → conflict detection across passages. The report
exposes each check separately; the gate (Step 8) applies the vetoes in order.

Thresholds come from ``configs/verification.yaml`` unless the registry supplies calibrated values
for the verifier model (``thresholds`` argument), which take precedence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.core.verdicts import ClaimVerdict, ConflictRecord, VerificationReport
from app.evidence.selector import EvidenceUnit
from app.generation.contract import GeneratedAnswer
from app.verification.base import Verifier
from app.verification.citations import index_by_citation, resolve
from app.verification.claims import prepare_claims
from app.verification.conflict import detect_conflict
from app.verification.entities import EntityExtractor, entity_consistency
from app.verification.evidence_map import PassageScore, score_claims
from app.verification.numeric import numeric_consistency


class VerificationConfig(BaseModel):
    entail_threshold: float = Field(0.60, ge=0, le=1)
    contradict_threshold: float = Field(0.55, ge=0, le=1)
    max_pairs_per_claim: int = Field(12, ge=1)
    numeric_rel_tolerance: float = Field(0.02, ge=0)
    numeric_abs_tolerance: float = Field(0.5, ge=0)
    conflict_min_support: float = Field(0.60, ge=0, le=1)
    conflict_min_contra: float = Field(0.55, ge=0, le=1)

    @classmethod
    def from_yaml(cls, path: Path | str) -> VerificationConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})

    def with_thresholds(self, thresholds: Mapping[str, float] | None) -> VerificationConfig:
        """Calibrated thresholds from the registry (keys 'entail', 'contradict') override YAML."""
        if not thresholds:
            return self
        upd = {}
        if "entail" in thresholds:
            upd["entail_threshold"] = float(thresholds["entail"])
        if "contradict" in thresholds:
            upd["contradict_threshold"] = float(thresholds["contradict"])
        return self.model_copy(update=upd)


def _label(scores: Sequence[PassageScore], cfg: VerificationConfig) -> tuple[str, PassageScore | None, PassageScore | None]:  # noqa: E501
    best_sup = max(scores, key=lambda p: p.scores.entail, default=None)
    best_con = max(scores, key=lambda p: p.scores.contradict, default=None)
    if best_con is not None and best_con.scores.contradict >= cfg.contradict_threshold:
        # contradiction first: a contradicted claim never counts as supported
        return "contradicted", best_sup, best_con
    if best_sup is not None and best_sup.scores.entail >= cfg.entail_threshold:
        return "supported", best_sup, best_con
    return "unsupported", best_sup, best_con


def verify(
    answer: GeneratedAnswer,
    evidence: Sequence[EvidenceUnit],
    verifier: Verifier,
    extractor: EntityExtractor,
    *,
    config: VerificationConfig | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> VerificationReport:
    cfg = (config or VerificationConfig()).with_thresholds(thresholds)
    trace: list[str] = []
    if answer.abstained:
        trace.append(f"generator abstained: {answer.abstain_reason}")
        return VerificationReport(verifier=verifier.name, policy_trace=trace)

    claims = prepare_claims(answer)
    by_cit = index_by_citation(evidence)
    scored = score_claims(claims, evidence, verifier, max_pairs=cfg.max_pairs_per_claim)
    verdicts: list[ClaimVerdict] = []
    conflicts: list[ConflictRecord] = []
    for c in claims:
        ps = scored.get(c.id, [])
        resolved, fabricated = resolve(c.citations, by_cit)
        label, best_sup, best_con = _label(ps, cfg)
        supporting_passages = [p for p in ps if p.scores.entail >= cfg.entail_threshold]
        supporting_ids = tuple(
            dict.fromkeys(p.unit.citation_id for p in supporting_passages if p.cited)
        )
        sup_texts = [p.unit.text for p in supporting_passages] or (
            [best_sup.unit.text] if best_sup else []
        )
        e_checked, e_ok, e_missing = entity_consistency(c.text, sup_texts, extractor)
        n_checked, n_ok, n_mism = numeric_consistency(
            c.text, sup_texts, rel_tol=cfg.numeric_rel_tolerance, abs_tol=cfg.numeric_abs_tolerance
        )
        conflict = detect_conflict(
            c.id, ps, min_support=cfg.conflict_min_support, min_contra=cfg.conflict_min_contra
        )
        if conflict:
            conflicts.append(conflict)
        notes = []
        if fabricated:
            notes.append(f"fabricated citation(s): {', '.join(fabricated)}")
        if e_missing:
            notes.append(f"entities not in supporting evidence: {', '.join(e_missing)}")
        if n_mism:
            notes.append(f"numbers not in supporting evidence: {', '.join(n_mism)}")
        verdicts.append(
            ClaimVerdict(
                claim_id=c.id,
                text=c.text,
                label=label,
                critical=c.critical,
                support=best_sup.scores.entail if best_sup else 0.0,
                contradiction=best_con.scores.contradict if best_con else 0.0,
                best_passage_id=best_sup.unit.passage_id if best_sup else None,
                contradicting_passage_id=(
                    best_con.unit.passage_id
                    if best_con and best_con.scores.contradict >= cfg.contradict_threshold
                    else None
                ),
                n_supporting=len(supporting_passages),
                n_contradicting=sum(p.scores.contradict >= cfg.contradict_threshold for p in ps),
                cited_ids=tuple(c.citations),
                resolved_ids=resolved,
                supporting_ids=supporting_ids,
                fabricated_ids=fabricated,
                entity_checked=e_checked,
                entity_ok=e_ok,
                entity_missing=e_missing,
                numeric_checked=n_checked,
                numeric_ok=n_ok,
                numeric_mismatches=n_mism,
                notes=tuple(notes),
            )
        )
        trace.append(
            f"{c.id}: {label} (support={verdicts[-1].support:.2f}, "
            f"contradiction={verdicts[-1].contradiction:.2f}"
            + (f", {'; '.join(notes)}" if notes else "")
            + ")"
        )
    report = VerificationReport.from_verdicts(verdicts, conflicts, verifier=verifier.name)
    report.policy_trace = trace + [f"conflict on {k.claim_id}: {k.description}" for k in conflicts]
    return report
