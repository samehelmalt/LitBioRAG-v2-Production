"""Evidence selection: turn a retrieval result into provenance-complete evidence units.

Nothing reaches the generator or the verifier except ``EvidenceUnit`` objects, and every unit
carries the identifiers and offsets needed to show, cite and re-check it. Preprints are flagged.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.core.documents import SourceType
from app.retrieval.pipeline import RetrievalResult
from app.retrieval.store import MetadataStore


class EvidenceUnit(BaseModel):
    passage_id: str
    doc_id: str
    pmid: str | None
    doi: str | None
    title: str
    year: int | None
    journal: str | None
    source_type: SourceType
    is_peer_reviewed: bool
    text: str
    char_start: int
    char_end: int
    score: float
    doc_rank: int
    url: str | None

    @property
    def citation_id(self) -> str:
        return f"PMID:{self.pmid}" if self.pmid else f"DOI:{self.doi}"


def select_evidence(
    result: RetrievalResult,
    store: MetadataStore,
    *,
    max_units: int = 10,
    per_doc: int = 2,
    max_docs: int | None = None,
) -> list[EvidenceUnit]:
    """Best ``per_doc`` passages of each top document, in document rank order, de-duplicated."""
    units: list[EvidenceUnit] = []
    seen_text: set[str] = set()
    scores = {p.passage_id: p.score for p in result.passages}
    docs = result.docs[:max_docs] if max_docs else result.docs
    for rank, cand in enumerate(docs, start=1):
        doc = store.get_document(cand.doc_id)
        if doc is None:
            continue
        passages = store.get_passages(cand.passage_ids[:per_doc])
        for pid in cand.passage_ids[:per_doc]:
            p = passages.get(pid)
            if p is None:
                continue
            key = p.text[:120].lower()
            if key in seen_text:
                continue
            seen_text.add(key)
            units.append(
                EvidenceUnit(
                    passage_id=p.passage_id,
                    doc_id=doc.doc_id,
                    pmid=doc.pmid,
                    doi=doc.doi,
                    title=doc.title,
                    year=doc.year,
                    journal=doc.journal,
                    source_type=doc.source_type,
                    is_peer_reviewed=doc.is_peer_reviewed,
                    text=p.text,
                    char_start=p.char_start,
                    char_end=p.char_end,
                    score=scores.get(pid, cand.score),
                    doc_rank=rank,
                    url=doc.url,
                )
            )
            if len(units) >= max_units:
                return units
    return units
