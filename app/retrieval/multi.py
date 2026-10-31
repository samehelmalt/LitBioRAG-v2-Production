"""Retrieval scope: the served corpus bundle (PubMed, bioRxiv/medRxiv), the caller's uploaded
knowledge bases, or both — merged by RRF and reranked once with the shared reranker.

Each source is a complete ``HybridRetriever`` (store + BM25 + dense). Entitlement filters are
applied inside each source before fusion: ``course_ids`` for the corpus bundle (course slides),
``kb_ids`` for the uploads bundle (``course_id`` column holds the kb id). A ``RoutedStore``
resolves document and passage ids to whichever source holds them, so evidence selection, the
API's source endpoint and the reviewer view work unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence

from app.core.documents import Document, Passage
from app.reranking.base import Candidate, Reranker
from app.retrieval.fusion import RankedItem, aggregate_to_documents, rrf
from app.retrieval.pipeline import (
    HybridRetriever,
    RankedPassage,
    RetrievalAudit,
    RetrievalConfig,
    RetrievalResult,
)
from app.retrieval.store import MetadataStore

SCOPES = ("indexed", "uploaded", "both")


class RoutedStore:
    """Read-only facade over several stores; ids are looked up in order (uploads first)."""

    def __init__(self, stores: Sequence[MetadataStore]):
        self.stores = list(stores)

    def get_document(self, doc_id: str) -> Document | None:
        for s in self.stores:
            d = s.get_document(doc_id)
            if d is not None:
                return d
        return None

    def get_passage(self, passage_id: str) -> Passage | None:
        for s in self.stores:
            p = s.get_passage(passage_id)
            if p is not None:
                return p
        return None

    def get_passages(self, passage_ids: Iterable[str]) -> dict[str, Passage]:
        want = list(passage_ids)
        out: dict[str, Passage] = {}
        for s in self.stores:
            missing = [p for p in want if p not in out]
            if not missing:
                break
            out.update(s.get_passages(missing))
        return out

    def passages_of(self, doc_id: str) -> list[Passage]:
        for s in self.stores:
            ps = s.passages_of(doc_id)
            if ps:
                return ps
        return []

    def course_of(self, doc_ids: Iterable[str]) -> dict[str, str | None]:
        want = list(doc_ids)
        out: dict[str, str | None] = {}
        for s in self.stores:
            out.update(s.course_of([d for d in want if d not in out]))
        return out


class MultiRetriever:
    def __init__(self, corpus: HybridRetriever | None, uploads: HybridRetriever | None,
                 reranker: Reranker | None, config: RetrievalConfig | None = None):  # fmt: skip
        if corpus is None and uploads is None:
            raise ValueError("at least one retrieval source is required")
        self.corpus = corpus
        self.uploads = uploads
        self.reranker = reranker
        self.cfg = config or (corpus or uploads).cfg  # type: ignore[union-attr]
        stores = [r.store for r in (uploads, corpus) if r is not None]
        self.store = RoutedStore(stores)

    def retrieve(
        self,
        query: str,
        *,
        scope: str = "both",
        course_ids: Sequence[str] | None = None,
        kb_ids: Sequence[str] | None = None,
    ) -> RetrievalResult:
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")
        t0 = time.perf_counter()
        audit = RetrievalAudit(mode=f"multi:{scope}")
        audit.course_filter = list(course_ids) if course_ids is not None else None
        lists: dict[str, list[RankedItem]] = {}
        if scope in ("indexed", "both") and self.corpus is not None:
            r = self.corpus.retrieve(query, mode="hybrid", course_ids=course_ids)
            lists["corpus"] = [RankedItem(p.passage_id, p.score) for p in r.passages]
            audit.query_symbols = r.audit.query_symbols
            for k, v in r.audit.timings_ms.items():
                audit.timings_ms[f"corpus.{k}"] = v
        if scope in ("uploaded", "both") and self.uploads is not None:
            # kb_ids = the caller's readable knowledge bases; [] means entitled to none
            r = self.uploads.retrieve(query, mode="hybrid", course_ids=list(kb_ids or []))
            lists["uploads"] = [RankedItem(p.passage_id, p.score) for p in r.passages]
            for k, v in r.audit.timings_ms.items():
                audit.timings_ms[f"uploads.{k}"] = v
        for name, items in lists.items():
            audit.list_sizes[name] = len(items)
        fused = rrf(lists, k=self.cfg.rrf_k) if len(lists) > 1 else None
        items = ([RankedItem(f.id, f.score) for f in fused] if fused is not None
                 else next(iter(lists.values()), []))  # fmt: skip
        pool = items[: self.cfg.rerank_pool]
        ranked = pool
        if self.reranker is not None and pool:
            t = time.perf_counter()
            texts = self.store.get_passages([p.id for p in pool])
            cands = [Candidate(p.id, texts[p.id].text) for p in pool if p.id in texts]
            hits = self.reranker.rerank(query, cands, top_n=len(cands))
            ranked = [RankedItem(h.passage_id, h.score) for h in hits]
            audit.timings_ms["rerank"] = (time.perf_counter() - t) * 1000
            audit.list_sizes["reranked"] = len(ranked)
        docs = aggregate_to_documents(ranked, max_passages_per_doc=self.cfg.passages_per_doc)
        docs = docs[: self.cfg.top_docs]
        for rnk, d in enumerate(docs, start=1):
            audit.doc_ranks.setdefault(d.doc_id, {})["final"] = rnk
        for name, lst in lists.items():
            seen: set[str] = set()
            for it in lst:
                did = it.id.rsplit("#", 1)[0]
                if did not in seen:
                    seen.add(did)
                    audit.doc_ranks.setdefault(did, {})[name] = len(seen)
        passages = [RankedPassage(p.id, p.id.rsplit("#", 1)[0], p.score) for p in ranked]
        audit.timings_ms["total"] = (time.perf_counter() - t0) * 1000
        return RetrievalResult(query=query, docs=docs, passages=passages, audit=audit)
