"""Ranking metrics over PMID lists.

Semantics follow the thesis implementation (``DualEvaluator._compute_ranking_metrics``,
engine_v5521.py L4474) so that production numbers are comparable with the imported CSVs:

* recall@k   = |gold ∩ top-k| / |gold|
* mrr        = 1 / rank of the first gold document in top-k (0 if none)
* ndcg@k     = binary-gain DCG / ideal DCG with min(|gold|, k) ideal hits
* *_norm     = the same, with gold restricted to documents present in the corpus
               (``None`` when no gold document is in the corpus)
* first_relevant_rank = 1-based rank of the first gold document in the full list (None if absent)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RankingResult:
    recall_at_k: float
    mrr: float
    ndcg: float
    recall_at_1: float
    first_relevant_rank: int | None
    n_gold: int
    n_gold_in_corpus: int
    recall_at_k_norm: float | None
    mrr_norm: float | None
    ndcg_norm: float | None


def _hits(ranked: Sequence[str], gold: set[str]) -> list[int]:
    return [1 if p in gold else 0 for p in ranked]


def _mrr(hits: Sequence[int]) -> float:
    for rank, h in enumerate(hits, start=1):
        if h:
            return 1.0 / rank
    return 0.0


def _ndcg(hits: Sequence[int], n_gold: int, k: int) -> float:
    dcg = sum(h / math.log2(r + 1) for r, h in enumerate(hits, start=1))
    n_ideal = min(n_gold, k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, n_ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    g = {str(x) for x in gold if str(x)}
    if not g:
        raise ValueError("gold set is empty")
    return len(g & set(ranked[:k])) / len(g)


def mrr(ranked: Sequence[str], gold: Iterable[str], k: int | None = None) -> float:
    g = {str(x) for x in gold if str(x)}
    return _mrr(_hits(ranked[:k] if k else ranked, g))


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    g = {str(x) for x in gold if str(x)}
    if not g:
        raise ValueError("gold set is empty")
    return _ndcg(_hits(ranked[:k], g), len(g), k)


def first_relevant_rank(ranked: Sequence[str], gold: Iterable[str]) -> int | None:
    g = {str(x) for x in gold if str(x)}
    for rank, p in enumerate(ranked, start=1):
        if p in g:
            return rank
    return None


def ranking_metrics(
    ranked: Sequence[str],
    gold: Iterable[str],
    k: int = 10,
    corpus_pmids: set[str] | None = None,
) -> RankingResult:
    """All metrics at once. ``ranked`` are PMIDs in rank order (already normalized)."""
    g = {str(x) for x in gold if str(x)}
    if not g:
        raise ValueError("gold set is empty")
    g_in = g & corpus_pmids if corpus_pmids is not None else set(g)
    top = list(ranked[:k])
    hits = _hits(top, g)
    res_norm: tuple[float | None, float | None, float | None]
    if g_in:
        hits_c = _hits(top, g_in)
        res_norm = (sum(hits_c) / len(g_in), _mrr(hits_c), _ndcg(hits_c, len(g_in), k))
    else:
        res_norm = (None, None, None)
    return RankingResult(
        recall_at_k=sum(hits) / len(g),
        mrr=_mrr(hits),
        ndcg=_ndcg(hits, len(g), k),
        recall_at_1=float(hits[0]) if hits else 0.0,
        first_relevant_rank=first_relevant_rank(ranked, g),
        n_gold=len(g),
        n_gold_in_corpus=len(g_in),
        recall_at_k_norm=res_norm[0],
        mrr_norm=res_norm[1],
        ndcg_norm=res_norm[2],
    )


def mean_or_none(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
