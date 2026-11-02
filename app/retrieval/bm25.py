"""On-disk BM25 over passages with Tantivy.

Two indexed views of every passage:
* ``text``       — English stemming tokenizer (recall on inflections),
* ``text_norm``  — pre-normalized surface form: lower-cased, hyphens/slashes/brackets split into
                   spaces, so that "Trifluridine/Tipiracil", "MAP3K3," and
                   "N6-methyladenosine (m6A)" match their parts exactly. Gene symbols are kept
                   intact as single tokens.
Queries are normalized the same way and parsed leniently, so user punctuation can never break
the parser. Scores are raw BM25; fusion normalizes them later.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import tantivy

from app.core.documents import Passage

_SPLIT = re.compile(r"[-/\\(\)\[\]{},;:\"'`<>=+*|]+")
_WS = re.compile(r"\s+")


def normalize_for_bm25(text: str) -> str:
    """Lower-case and split punctuation that glues biomedical tokens together."""
    t = _SPLIT.sub(" ", text.lower())
    t = re.sub(r"\.(?=\s|$)", " ", t)  # sentence-final periods; decimals are untouched
    return _WS.sub(" ", t).strip()


@dataclass(frozen=True)
class BM25Hit:
    passage_id: str
    doc_id: str
    score: float


def _schema() -> tantivy.Schema:
    sb = tantivy.SchemaBuilder()
    sb.add_text_field("passage_id", stored=True, tokenizer_name="raw")
    sb.add_text_field("doc_id", stored=True, tokenizer_name="raw")
    sb.add_text_field("source_type", stored=True, tokenizer_name="raw")
    sb.add_text_field("course_id", stored=True, tokenizer_name="raw")  # entitlement filter
    sb.add_integer_field("year", stored=True, indexed=True, fast=True)
    sb.add_text_field("text", stored=False, tokenizer_name="en_stem")
    sb.add_text_field("text_norm", stored=False, tokenizer_name="default")
    return sb.build()


class BM25Index:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.index = tantivy.Index(_schema(), path=str(self.path))
        self._searcher: tantivy.Searcher | None = None

    # ---- build ------------------------------------------------------------------------
    def add(
        self,
        passages: Iterable[Passage],
        meta: dict[str, tuple] | None = None,
        *,
        heap_mb: int = 256,
        commit_every: int = 50_000,
    ) -> int:
        """Index passages; ``meta`` maps doc_id -> (year, source_type[, course_id])."""
        writer = self.index.writer(heap_size=heap_mb * 1024 * 1024)
        n = 0
        for p in passages:
            m = (meta or {}).get(p.doc_id, (None, ""))
            year, st = m[0], m[1]
            course = m[2] if len(m) > 2 and m[2] else ""
            fields = {
                "passage_id": p.passage_id,
                "doc_id": p.doc_id,
                "source_type": st or "",
                "course_id": course or "-",
                "text": p.text,
                "text_norm": normalize_for_bm25(p.text),
            }
            if year is not None:
                fields["year"] = int(year)
            writer.add_document(tantivy.Document(**fields))
            n += 1
            if n % commit_every == 0:
                writer.commit()
        writer.commit()
        writer.wait_merging_threads()
        self.index.reload()
        self._searcher = None
        return n

    # ---- search -----------------------------------------------------------------------
    def delete(self, passage_ids: Iterable[str], *, heap_mb: int = 64) -> int:
        """Remove passages by id (Tantivy term delete); used by the incremental KB indexer."""
        ids = list(passage_ids)
        if not ids:
            return 0
        writer = self.index.writer(heap_size=heap_mb * 1024 * 1024)
        for pid in ids:
            writer.delete_documents_by_term("passage_id", pid)
        writer.commit()
        writer.wait_merging_threads()
        self.index.reload()
        self._searcher = None
        return len(ids)

    def _get_searcher(self) -> tantivy.Searcher:
        if self._searcher is None:
            self.index.reload()
            self._searcher = self.index.searcher()
        return self._searcher

    def num_docs(self) -> int:
        return self._get_searcher().num_docs

    @staticmethod
    def _course_clause(course_ids: Sequence[str] | None) -> str:
        """'+(course_id:a OR course_id:b) ' — entitlement filter applied inside the engine."""
        if course_ids is None:
            return ""
        ids = [re.sub(r"[^A-Za-z0-9_.-]", "", c) for c in course_ids if c]
        if not ids:
            return "+course_id:__none__ "  # entitled to nothing → nothing matches
        return "+(" + " OR ".join(f"course_id:{c}" for c in ids) + ") "

    def search(
        self, query: str, k: int = 100, *, course_ids: Sequence[str] | None = None
    ) -> list[BM25Hit]:
        q_norm = normalize_for_bm25(query)
        if not q_norm:
            return []
        searcher = self._get_searcher()
        q_text = self._course_clause(course_ids) + q_norm
        q = self.index.parse_query_lenient(q_text, ["text", "text_norm"])[0]
        hits: list[BM25Hit] = []
        for score, addr in searcher.search(q, k).hits:
            d = searcher.doc(addr)
            hits.append(BM25Hit(d["passage_id"][0], d["doc_id"][0], float(score)))
        return hits

    def search_many(self, queries: Sequence[str], k: int = 100) -> list[list[BM25Hit]]:
        return [self.search(q, k) for q in queries]

    def search_must(
        self, symbols: Sequence[str], k: int = 100, *, course_ids: Sequence[str] | None = None
    ) -> list[BM25Hit]:
        """Passages containing EVERY given symbol literally (normalized view). Used as the
        exact-match channel for gene symbols and codes; near neighbours never match."""
        terms = [normalize_for_bm25(s) for s in symbols if normalize_for_bm25(s)]
        if not terms:
            return []
        q_text = self._course_clause(course_ids) + " ".join(
            f"+text_norm:{t.split()[0]}" for t in terms if t.split()
        )
        searcher = self._get_searcher()
        q = self.index.parse_query_lenient(q_text, ["text_norm"])[0]
        hits: list[BM25Hit] = []
        for score, addr in searcher.search(q, k).hits:
            d = searcher.doc(addr)
            hits.append(BM25Hit(d["passage_id"][0], d["doc_id"][0], float(score)))
        return hits
