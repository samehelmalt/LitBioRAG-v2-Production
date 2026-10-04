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
        meta: dict[str, tuple[int | None, str]] | None = None,
        *,
        heap_mb: int = 256,
        commit_every: int = 50_000,
    ) -> int:
        """Index passages; ``meta`` maps doc_id -> (year, source_type) for filter fields."""
        writer = self.index.writer(heap_size=heap_mb * 1024 * 1024)
        n = 0
        for p in passages:
            year, st = (meta or {}).get(p.doc_id, (None, ""))
            fields = {
                "passage_id": p.passage_id,
                "doc_id": p.doc_id,
                "source_type": st or "",
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
    def _get_searcher(self) -> tantivy.Searcher:
        if self._searcher is None:
            self.index.reload()
            self._searcher = self.index.searcher()
        return self._searcher

    def num_docs(self) -> int:
        return self._get_searcher().num_docs

    def search(self, query: str, k: int = 100) -> list[BM25Hit]:
        q_norm = normalize_for_bm25(query)
        if not q_norm:
            return []
        searcher = self._get_searcher()
        q = self.index.parse_query_lenient(q_norm, ["text", "text_norm"])[0]
        hits: list[BM25Hit] = []
        for score, addr in searcher.search(q, k).hits:
            d = searcher.doc(addr)
            hits.append(BM25Hit(d["passage_id"][0], d["doc_id"][0], float(score)))
        return hits

    def search_many(self, queries: Sequence[str], k: int = 100) -> list[list[BM25Hit]]:
        return [self.search(q, k) for q in queries]
