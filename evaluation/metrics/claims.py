"""Claim-level aggregates.

The verification pipeline (app/verification) emits one ``ClaimVerdict`` per atomic claim. These
functions turn a list of verdicts into the generation metrics reported in BASELINE_RESULTS.md.
They are deliberately model-free: whatever verifier is chosen, the aggregation is the same.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

# One definition of a verdict for runtime and evaluation (app/core/verdicts.py). Evaluation may
# import from app; app never imports from evaluation.
from app.core.verdicts import LABELS, ClaimVerdict

__all__ = ["LABELS", "ClaimSummary", "ClaimVerdict", "summarize"]


@dataclass(frozen=True)
class ClaimSummary:
    n_claims: int
    entailment_rate: float | None
    unsupported_rate: float | None
    contradiction_rate: float | None
    critical_contradicted: int
    citation_resolution_rate: float | None  # cited ids that exist in the retrieved set
    citation_support_rate: float | None  # cited ids that exist AND support the claim
    uncited_claim_rate: float | None
    numeric_fidelity: float | None  # None when no numeric values were checked
    entity_fidelity: float | None
    notes: tuple[str, ...] = field(default_factory=tuple)


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def summarize(verdicts: Iterable[ClaimVerdict]) -> ClaimSummary:
    vs = list(verdicts)
    n = len(vs)
    if n == 0:
        return ClaimSummary(0, None, None, None, 0, None, None, None, None, None, ("no claims",))
    by = {lab: sum(v.label == lab for v in vs) for lab in LABELS}
    cited = sum(len(v.cited_ids) for v in vs)
    resolved = sum(len(v.resolved_ids) for v in vs)
    supporting = sum(len(v.supporting_ids) for v in vs)
    num_c = sum(v.numeric_checked for v in vs)
    num_ok = sum(v.numeric_ok for v in vs)
    ent_c = sum(v.entity_checked for v in vs)
    ent_ok = sum(v.entity_ok for v in vs)
    return ClaimSummary(
        n_claims=n,
        entailment_rate=by["supported"] / n,
        unsupported_rate=by["unsupported"] / n,
        contradiction_rate=by["contradicted"] / n,
        critical_contradicted=sum(v.critical and v.label == "contradicted" for v in vs),
        citation_resolution_rate=_ratio(resolved, cited),
        citation_support_rate=_ratio(supporting, cited),
        uncited_claim_rate=sum(not v.cited_ids for v in vs) / n,
        numeric_fidelity=_ratio(num_ok, num_c),
        entity_fidelity=_ratio(ent_ok, ent_c),
    )
