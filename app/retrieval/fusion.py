"""Reciprocal rank fusion and passage -> document aggregation.

RRF (Cormack et al., 2009): score(item) = Σ_lists w_list / (k + rank_in_list). It needs no score
calibration between BM25 and dense lists, which is why the thesis used it (engine_v5521.py
``rrf_fusion``). The thesis's confidence-based re-weighting wrapper is deliberately not ported: its
input signal did not discriminate (PRODUCTION_AUDIT.md §9) and weights are a benchmark parameter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RankedItem:
    id: str
    score: float


@dataclass
class FusedItem:
    id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # list name -> 1-based rank


def rrf(
    lists: Mapping[str, Sequence[RankedItem]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[FusedItem]:
    """Fuse named ranked lists. Items absent from a list contribute nothing from it."""
    if k <= 0:
        raise ValueError("k must be positive")
    weights = dict(weights or {})
    fused: dict[str, FusedItem] = {}
    for name, items in lists.items():
        w = float(weights.get(name, 1.0))
        for rank, it in enumerate(items, start=1):
            f = fused.get(it.id)
            if f is None:
                f = fused[it.id] = FusedItem(it.id, 0.0)
            if name in f.ranks:  # duplicate id inside one list: keep the first (best) rank
                continue
            f.ranks[name] = rank
            f.score += w / (k + rank)
    return sorted(fused.values(), key=lambda f: (-f.score, f.id))


def doc_id_of(passage_id: str) -> str:
    return passage_id.rsplit("#", 1)[0]


@dataclass
class DocCandidate:
    doc_id: str
    score: float
    passage_ids: list[str]  # best passage first


def aggregate_to_documents(
    passages: Sequence[RankedItem | FusedItem], *, max_passages_per_doc: int = 3
) -> list[DocCandidate]:
    """Collapse a passage ranking to a document ranking (max passage score per document)."""
    docs: dict[str, DocCandidate] = {}
    for p in passages:
        d = doc_id_of(p.id)
        c = docs.get(d)
        if c is None:
            docs[d] = DocCandidate(d, p.score, [p.id])
        else:
            if len(c.passage_ids) < max_passages_per_doc:
                c.passage_ids.append(p.id)
            c.score = max(c.score, p.score)
    return sorted(docs.values(), key=lambda c: (-c.score, c.doc_id))
