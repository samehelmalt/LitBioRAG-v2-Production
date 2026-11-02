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
    # course-material provenance (slides)
    course_id: str | None = None
    lecture_id: str | None = None
    lecture_title: str | None = None
    slide_number: int | None = None
    neighbour_of: str | None = None  # passage id of the slide this unit was added next to
    # uploaded-file provenance (knowledge bases)
    kb_id: str | None = None
    file_id: str | None = None
    file_name: str | None = None
    locator: str | None = None  # "p3" | "s7" | "Sheet1!A1:D40" | "sec2"
    locator_label: str | None = None  # "page 3" | "slide 7" | "Sheet1 rows 1–40"
    canonical_id: str | None = None  # course/lecture/slide/version/hash for uploaded decks
    deck_version: str | None = None
    superseded_by: str | None = None

    @property
    def is_slide(self) -> bool:
        return self.doc_id.startswith("slide:")

    @property
    def is_file(self) -> bool:
        return self.doc_id.startswith("file:")

    @property
    def is_lecture_slide(self) -> bool:
        return bool(self.lecture_id) and self.slide_number is not None

    @property
    def citation_id(self) -> str:
        if self.is_slide or (self.is_file and self.is_lecture_slide):
            return f"SLIDE:{self.lecture_id}:{self.slide_number}"
        if self.is_file:
            return f"FILE:{self.file_id}:{self.locator}"
        return f"PMID:{self.pmid}" if self.pmid else f"DOI:{self.doi}"

    @property
    def display_source(self) -> str:
        """'Lecture title, Slide N (v2026)' | 'file name, page N' | the article title."""
        if self.is_slide or (self.is_file and self.is_lecture_slide):
            v = f" (v{self.deck_version})" if self.deck_version else ""
            return f"{self.lecture_title}, Slide {self.slide_number}{v}"
        if self.is_file:
            return f"{self.file_name}, {self.locator_label or self.locator}"
        return self.title


def select_evidence(
    result: RetrievalResult,
    store: MetadataStore,
    *,
    max_units: int = 10,
    per_doc: int = 2,
    max_docs: int | None = None,
    slide_neighbours: int = 1,
    course_share: float = 0.0,
) -> list[EvidenceUnit]:
    """Best ``per_doc`` passages of each top document, in document rank order, de-duplicated.

    For lecture slides the definition is often on slide n−1 and the exception on slide n+1, so the
    ``slide_neighbours`` adjacent slides of the same lecture are appended after a matched slide
    (marked ``neighbour_of``); they count toward ``max_units``.
    """
    if course_share > 0:
        # lecture-first: over-select, then guarantee the course-material share
        wide = select_evidence(result, store, max_units=max_units * 3, per_doc=per_doc,
                               max_docs=max_docs, slide_neighbours=slide_neighbours)  # fmt: skip
        return _rebalance(wide, max_units, course_share)
    units: list[EvidenceUnit] = []
    seen_text: set[str] = set()
    seen_docs: set[str] = set()
    scores = {p.passage_id: p.score for p in result.passages}
    docs = result.docs[:max_docs] if max_docs else result.docs

    def _unit(doc, p, rank, score, neighbour_of=None) -> EvidenceUnit:
        s = doc.slide
        f = doc.file
        return EvidenceUnit(
            passage_id=p.passage_id, doc_id=doc.doc_id, pmid=doc.pmid, doi=doc.doi,
            title=doc.title, year=doc.year, journal=doc.journal, source_type=doc.source_type,
            is_peer_reviewed=doc.is_peer_reviewed, text=p.text, char_start=p.char_start,
            char_end=p.char_end, score=score, doc_rank=rank, url=doc.url,
            course_id=(s.course_id if s else (f.course_id or f.kb_id) if f else None),
            lecture_id=(s.lecture_id if s else (f.lecture_id or None) if f else None),
            lecture_title=(s.lecture_title if s else (f.lecture_title or None) if f else None),
            slide_number=(s.slide_number if s else f.slide_number if f else None),
            neighbour_of=neighbour_of,
            kb_id=f.kb_id if f else None, file_id=f.file_id if f else None,
            file_name=f.file_name if f else None, locator=f.locator if f else None,
            locator_label=f.label if f else None, canonical_id=f.canonical_id if f else None,
            deck_version=(f.deck_version or None) if f else None,
            superseded_by=(f.superseded_by or None) if f else None,
        )  # fmt: skip

    for rank, cand in enumerate(docs, start=1):
        doc = store.get_document(cand.doc_id)
        if doc is None:
            continue
        seen_docs.add(doc.doc_id)
        passages = store.get_passages(cand.passage_ids[:per_doc])
        added_slide = None
        for pid in cand.passage_ids[:per_doc]:
            p = passages.get(pid)
            if p is None:
                continue
            key = p.text[:120].lower()
            if key in seen_text:
                continue
            seen_text.add(key)
            units.append(_unit(doc, p, rank, scores.get(pid, cand.score)))
            added_slide = added_slide or (p.passage_id if doc.slide else None)
            if len(units) >= max_units:
                return units
        if added_slide and slide_neighbours > 0:
            s = doc.slide
            for delta in range(1, slide_neighbours + 1):
                for n in (s.slide_number - delta, s.slide_number + delta):
                    nid = doc.make_slide_id(s.lecture_id, n)
                    if n < 1 or nid in seen_docs:
                        continue
                    ndoc = store.get_document(nid)
                    if ndoc is None:
                        continue
                    nps = store.passages_of(nid)
                    if not nps:
                        continue
                    seen_docs.add(nid)
                    seen_text.add(nps[0].text[:120].lower())
                    units.append(_unit(ndoc, nps[0], rank, cand.score * 0.5, added_slide))
                    if len(units) >= max_units:
                        return units
    return units


def _rebalance(units: list[EvidenceUnit], max_units: int,
               course_share: float) -> list[EvidenceUnit]:  # fmt: skip
    """Keep rank order but reserve at least ``course_share`` of the slots for course material
    when enough of it was retrieved; literature fills the rest."""
    import math

    course = [u for u in units if u.is_slide or u.is_lecture_slide]
    other = [u for u in units if not (u.is_slide or u.is_lecture_slide)]
    want_course = min(len(course), math.ceil(max_units * course_share))
    chosen = set(id(u) for u in course[:want_course])
    for u in other:
        if len(chosen) >= max_units:
            break
        chosen.add(id(u))
    for u in course[want_course:]:
        if len(chosen) >= max_units:
            break
        chosen.add(id(u))
    return [u for u in units if id(u) in chosen][:max_units]
