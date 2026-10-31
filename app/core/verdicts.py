"""Verification outputs shared by runtime and offline evaluation.

Each check reports its own field; nothing is folded into a composite here. The gate (Step 8)
reads these fields in a fixed order of vetoes and the policy trace records which rule decided.
``evaluation/metrics/claims.py`` aggregates the same objects, so runtime and evaluation can never
disagree about what a verdict means.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

LABELS = ("supported", "unsupported", "contradicted")


class ClaimVerdict(BaseModel):
    claim_id: str = ""
    text: str = ""
    label: str = "unsupported"
    critical: bool = False
    # entailment / contradiction (max over evidence passages)
    support: float = 0.0
    contradiction: float = 0.0
    best_passage_id: str | None = None
    contradicting_passage_id: str | None = None
    n_supporting: int = 0
    n_contradicting: int = 0
    # citations
    cited_ids: tuple[str, ...] = ()
    resolved_ids: tuple[str, ...] = ()  # cited ids present in the evidence set
    supporting_ids: tuple[str, ...] = ()  # resolved ids whose passage entails the claim
    fabricated_ids: tuple[str, ...] = ()  # cited ids absent from the evidence set
    # entity / numeric consistency against the supporting evidence
    entity_checked: int = 0
    entity_ok: int = 0
    entity_missing: tuple[str, ...] = ()
    numeric_checked: int = 0
    numeric_ok: int = 0
    numeric_mismatches: tuple[str, ...] = ()
    # question-level grounding (critical claims only)
    query_entities_missing: tuple[str, ...] = ()  # question entities this claim should name
    near_neighbour_substitution: bool = False  # e.g. MAP3K7 where the question asked MAP3K3
    substituted_for: str | None = None  # "map3k7->map3k3"
    notes: tuple[str, ...] = ()

    def model_post_init(self, __context) -> None:  # noqa: D105
        if self.label not in LABELS:
            raise ValueError(f"unknown label {self.label!r}")

    @property
    def entity_consistent(self) -> bool | None:
        return None if self.entity_checked == 0 else self.entity_ok == self.entity_checked

    @property
    def numeric_consistent(self) -> bool | None:
        return None if self.numeric_checked == 0 else self.numeric_ok == self.numeric_checked

    @property
    def citations_consistent(self) -> bool | None:
        if not self.cited_ids:
            return None
        return not self.fabricated_ids and bool(self.supporting_ids)


class ConflictRecord(BaseModel):
    claim_id: str
    supporting_ids: tuple[str, ...] = ()
    contradicting_ids: tuple[str, ...] = ()
    description: str = ""


class VerificationReport(BaseModel):
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    # separate scores; never combined here
    n_claims: int = 0
    entailment_rate: float | None = None
    unsupported_rate: float | None = None
    contradiction_rate: float | None = None
    critical_contradicted: int = 0
    critical_unsupported: int = 0
    citation_resolution_rate: float | None = None
    citation_support_rate: float | None = None
    uncited_claim_rate: float | None = None
    fabricated_citations: tuple[str, ...] = ()
    entity_fidelity: float | None = None
    numeric_fidelity: float | None = None
    retrieval_sufficiency: float | None = None  # filled by the evidence-sufficiency model (Step 8)
    question_entities: tuple[str, ...] = ()
    query_entity_grounding: float | None = None  # share of question entities in supporting evidence
    question_entities_missing_in_claims: tuple[str, ...] = ()
    substitution_detected: bool = False
    verifier: str = ""
    policy_trace: list[str] = Field(default_factory=list)

    @staticmethod
    def from_verdicts(verdicts: list[ClaimVerdict], conflicts: list[ConflictRecord] | None = None,
                      *, verifier: str = "") -> VerificationReport:  # fmt: skip
        n = len(verdicts)
        if n == 0:
            return VerificationReport(verifier=verifier, conflicts=conflicts or [])

        def ratio(num: int, den: int) -> float | None:
            return num / den if den else None

        cited = sum(len(v.cited_ids) for v in verdicts)
        resolved = sum(len(v.resolved_ids) for v in verdicts)
        supporting = sum(len(v.supporting_ids) for v in verdicts)
        fab: list[str] = []
        for v in verdicts:
            for f in v.fabricated_ids:
                if f not in fab:
                    fab.append(f)
        return VerificationReport(
            verdicts=verdicts,
            conflicts=conflicts or [],
            n_claims=n,
            entailment_rate=sum(v.label == "supported" for v in verdicts) / n,
            unsupported_rate=sum(v.label == "unsupported" for v in verdicts) / n,
            contradiction_rate=sum(v.label == "contradicted" for v in verdicts) / n,
            critical_contradicted=sum(v.critical and v.label == "contradicted" for v in verdicts),
            critical_unsupported=sum(v.critical and v.label == "unsupported" for v in verdicts),
            citation_resolution_rate=ratio(resolved, cited),
            citation_support_rate=ratio(supporting, cited),
            uncited_claim_rate=sum(not v.cited_ids for v in verdicts) / n,
            fabricated_citations=tuple(fab),
            entity_fidelity=ratio(sum(v.entity_ok for v in verdicts),
                                  sum(v.entity_checked for v in verdicts)),  # fmt: skip
            numeric_fidelity=ratio(sum(v.numeric_ok for v in verdicts),
                                   sum(v.numeric_checked for v in verdicts)),  # fmt: skip
            substitution_detected=any(v.near_neighbour_substitution for v in verdicts),
            verifier=verifier,
        )
