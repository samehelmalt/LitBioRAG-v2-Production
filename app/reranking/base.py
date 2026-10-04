"""Reranker protocol and a dependency-free implementation for tests.

A reranker scores (query, passage text) pairs over the *whole* candidate pool in one stage. There
is no cascade here by design: the thesis cascade pre-filtered with the retrieval encoder itself
and lost recall (PRODUCTION_AUDIT.md §2). If a two-stage design is ever re-introduced, stage one
must be a different, cheaper model and the gain must be shown by the Step 5 benchmark.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Candidate:
    passage_id: str
    text: str


@dataclass(frozen=True)
class RerankHit:
    passage_id: str
    score: float


class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, candidates: Sequence[Candidate], top_n: int
    ) -> list[RerankHit]: ...


class LexicalOverlapReranker:
    """Deterministic token-overlap scorer (query terms covered, weighted by rarity in the pool).

    Used in tests and CI so the pipeline runs without model weights. Never used for reported
    numbers.
    """

    name = "lexical-overlap"
    _tok = re.compile(r"[a-z0-9]+")

    def rerank(self, query: str, candidates: Sequence[Candidate], top_n: int) -> list[RerankHit]:
        q = set(self._tok.findall(query.lower())) - {"the", "a", "an", "of", "in", "is", "what"}
        if not q or not candidates:
            return []
        docs_tok = [set(self._tok.findall(c.text.lower())) for c in candidates]
        n = len(candidates)
        df = {t: sum(t in d for d in docs_tok) for t in q}
        hits = []
        for c, toks in zip(candidates, docs_tok, strict=True):
            s = sum((1.0 + (n / (1 + df[t]))) for t in q if t in toks)
            hits.append(RerankHit(c.passage_id, float(s)))
        hits.sort(key=lambda h: (-h.score, h.passage_id))
        return hits[:top_n]
