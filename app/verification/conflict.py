"""Evidence conflict: a claim that one source supports and another contradicts.

The thesis reduced evidence to supported / unsupported / contradicted per claim and the SBE prompt
told the model to follow the majority. Here conflict is a first-class record that the gate turns
into "the evidence is mixed" rather than a verdict for one side.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.verdicts import ConflictRecord
from app.verification.evidence_map import PassageScore


def tier_of(unit) -> str:
    """course_material for lecture slides (bundled or uploaded decks), literature otherwise."""
    if getattr(unit, "is_slide", False) or getattr(unit, "is_lecture_slide", False):
        return "course_material"
    return "literature"


def detect_conflict(
    claim_id: str,
    scores: Sequence[PassageScore],
    *,
    min_support: float,
    min_contra: float,
) -> ConflictRecord | None:
    sup = [p for p in scores if p.scores.entail >= min_support]
    con = [p for p in scores if p.scores.contradict >= min_contra]
    sup_ids = tuple(dict.fromkeys(p.unit.citation_id for p in sup))
    con_ids = tuple(dict.fromkeys(p.unit.citation_id for p in con))
    if not sup_ids or not con_ids or set(sup_ids) == set(con_ids):
        return None
    years = sorted({p.unit.year for p in sup + con if p.unit.year})
    kinds = sorted({p.unit.source_type.value for p in sup + con})
    sup_tiers = tuple(sorted({tier_of(p.unit) for p in sup}))
    con_tiers = tuple(sorted({tier_of(p.unit) for p in con}))
    lecture_vs_lit = (
        (sup_tiers == ("course_material",) and con_tiers == ("literature",))
        or (sup_tiers == ("literature",) and con_tiers == ("course_material",))
    )
    notes = []
    if len(years) > 1:
        notes.append(f"publication years {years[0]}–{years[-1]}")
    if len(kinds) > 1:
        notes.append("study types " + ", ".join(kinds))
    if lecture_vs_lit:
        desc = ("the course material and the literature differ on this claim"
                + (f" ({'; '.join(notes)})" if notes else ""))
    else:
        desc = (
            f"{len(sup_ids)} source(s) support and {len(con_ids)} contradict this claim"
            + (f" ({'; '.join(notes)})" if notes else "")
        )
    return ConflictRecord(
        claim_id=claim_id, supporting_ids=sup_ids, contradicting_ids=con_ids, description=desc,
        kind="lecture_vs_literature" if lecture_vs_lit else "evidence",
        supporting_tiers=sup_tiers, contradicting_tiers=con_tiers
    )
