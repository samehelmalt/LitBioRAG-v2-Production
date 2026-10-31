"""Citation resolution: a cited id must exist in the evidence set, and its passage must support
the claim. Ids that are not in the evidence set are fabricated by definition — the generator never
saw them — and fail regardless of how plausible they look."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from app.evidence.selector import EvidenceUnit


def index_by_citation(units: Iterable[EvidenceUnit]) -> dict[str, list[EvidenceUnit]]:
    idx: dict[str, list[EvidenceUnit]] = {}
    for u in units:
        idx.setdefault(u.citation_id, []).append(u)
    return idx


def resolve(
    cited: Iterable[str], by_cit: Mapping[str, list[EvidenceUnit]]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (resolved, fabricated) in citation order."""
    resolved, fabricated = [], []
    for c in cited:
        (resolved if c in by_cit else fabricated).append(c)
    return tuple(resolved), tuple(fabricated)
