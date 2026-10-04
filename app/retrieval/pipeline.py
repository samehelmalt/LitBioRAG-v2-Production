"""Hybrid retrieval: BM25 ‖ dense -> RRF -> single-stage reranker -> documents + passages.

Modes let the benchmark switch one component at a time:
    bm25 | dense | hybrid (RRF, no reranker) | hybrid+rerank
Every result carries an audit with the rank of each candidate document in each stage, so a missed
gold document can be attributed to the stage that lost it (RETRIEVAL_FAILURE_ANALYSIS.md).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.reranking.base import Candidate, Reranker
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.fusion import DocCandidate, RankedItem, aggregate_to_documents, rrf
from app.retrieval.store import MetadataStore

MODES = ("bm25", "dense", "hybrid", "hybrid+rerank")


class RetrievalConfig(BaseModel):
    k_bm25: int = Field(200, ge=1)
    k_dense: int = Field(200, ge=1)
    rrf_k: int = Field(60, ge=1)
    w_bm25: float = Field(0.5, ge=0)
    w_dense: float = Field(0.5, ge=0)
    rerank_pool: int = Field(100, ge=1)
    top_docs: int = Field(20, ge=1)
    passages_per_doc: int = Field(3, ge=1)

    @classmethod
    def from_yaml(cls, path: Path | str) -> RetrievalConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})


@dataclass
class RankedPassage:
    passage_id: str
    doc_id: str
    score: float


@dataclass
class RetrievalAudit:
    mode: str
    list_sizes: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    # doc_id -> {stage: 1-based rank}; stages: bm25, dense, fused, reranked, final
    doc_ranks: dict[str, dict[str, int]] = field(default_factory=dict)

    def rank_of(self, doc_id: str, stage: str) -> int | None:
        return self.doc_ranks.get(doc_id, {}).get(stage)


@dataclass
class RetrievalResult:
    query: str
    docs: list[DocCandidate]  # final document ranking
    passages: list[RankedPassage]  # final passage ranking (pool after rerank, or fused)
    audit: RetrievalAudit


def _doc_rank_map(items: Sequence[RankedItem]) -> dict[str, int]:
    return {d.doc_id: r for r, d in enumerate(aggregate_to_documents(items), start=1)}


class HybridRetriever:
    def __init__(
        self,
        store: MetadataStore,
        bm25: BM25Index | None,
        dense: DenseIndex | None,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ):
        self.store = store
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.cfg = config or RetrievalConfig()

    def retrieve(self, query: str, mode: str = "hybrid+rerank") -> RetrievalResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        cfg = self.cfg
        audit = RetrievalAudit(mode=mode)
        lists: dict[str, list[RankedItem]] = {}

        if mode in ("bm25", "hybrid", "hybrid+rerank"):
            if self.bm25 is None:
                raise RuntimeError("BM25 index not loaded")
            t = time.perf_counter()
            hits = self.bm25.search(query, k=cfg.k_bm25)
            lists["bm25"] = [RankedItem(h.passage_id, h.score) for h in hits]
            audit.timings_ms["bm25"] = (time.perf_counter() - t) * 1000
        if mode in ("dense", "hybrid", "hybrid+rerank"):
            if self.dense is None:
                raise RuntimeError("dense index not loaded")
            t = time.perf_counter()
            hits = self.dense.search(query, k=cfg.k_dense)
            lists["dense"] = [RankedItem(h.passage_id, h.score) for h in hits]
            audit.timings_ms["dense"] = (time.perf_counter() - t) * 1000
        for name, items in lists.items():
            audit.list_sizes[name] = len(items)
            for d, r in _doc_rank_map(items).items():
                audit.doc_ranks.setdefault(d, {})[name] = r

        if mode in ("bm25", "dense"):
            fused_items = lists[mode]
        else:
            t = time.perf_counter()
            fused = rrf(lists, k=cfg.rrf_k, weights={"bm25": cfg.w_bm25, "dense": cfg.w_dense})
            fused_items = [RankedItem(f.id, f.score) for f in fused]
            audit.timings_ms["fusion"] = (time.perf_counter() - t) * 1000
            audit.list_sizes["fused"] = len(fused_items)
            for d, r in _doc_rank_map(fused_items).items():
                audit.doc_ranks.setdefault(d, {})["fused"] = r

        pool = fused_items[: cfg.rerank_pool]
        ranked = pool
        if mode == "hybrid+rerank":
            if self.reranker is None:
                raise RuntimeError("mode hybrid+rerank needs a reranker")
            t = time.perf_counter()
            texts = self.store.get_passages([p.id for p in pool])
            cands = [Candidate(p.id, texts[p.id].text) for p in pool if p.id in texts]
            hits = self.reranker.rerank(query, cands, top_n=len(cands))
            ranked = [RankedItem(h.passage_id, h.score) for h in hits]
            audit.timings_ms["rerank"] = (time.perf_counter() - t) * 1000
            audit.list_sizes["reranked"] = len(ranked)
            for d, r in _doc_rank_map(ranked).items():
                audit.doc_ranks.setdefault(d, {})["reranked"] = r

        docs = aggregate_to_documents(ranked, max_passages_per_doc=cfg.passages_per_doc)
        docs = docs[: cfg.top_docs]
        for r, d in enumerate(docs, start=1):
            audit.doc_ranks.setdefault(d.doc_id, {})["final"] = r
        passages = [RankedPassage(p.id, p.id.rsplit("#", 1)[0], p.score) for p in ranked]
        audit.timings_ms["total"] = sum(v for k, v in audit.timings_ms.items() if k != "total")
        return RetrievalResult(query=query, docs=docs, passages=passages, audit=audit)
