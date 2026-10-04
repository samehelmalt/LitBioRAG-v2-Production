"""SQLite metadata store for documents and passages.

One file inside the artifact bundle (``metadata.sqlite``). Written on the build host, read-only on
the serving host. Lookups: by doc_id, PMID, DOI, passage_id. No text search here; that is the job
of the BM25 and dense indexes, which return passage ids that this store resolves to records.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path

from app.core.documents import Document, Passage, SourceType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    pmid        TEXT,
    doi         TEXT,
    title       TEXT NOT NULL,
    abstract    TEXT NOT NULL,
    source_type TEXT NOT NULL,
    year        INTEGER,
    journal     TEXT,
    pub_types   TEXT NOT NULL DEFAULT '[]',
    mesh_terms  TEXT NOT NULL DEFAULT '[]',
    authors     TEXT NOT NULL DEFAULT '[]',
    url         TEXT,
    source      TEXT NOT NULL,
    version     TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_pmid ON documents(pmid);
CREATE INDEX IF NOT EXISTS idx_documents_doi  ON documents(doi);
CREATE TABLE IF NOT EXISTS passages (
    passage_id TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL REFERENCES documents(doc_id),
    ord        INTEGER NOT NULL,
    text       TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end   INTEGER NOT NULL,
    n_words    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_passages_doc ON passages(doc_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class MetadataStore:
    def __init__(self, path: Path | str, *, read_only: bool = False):
        self.path = Path(path)
        if read_only:
            uri = f"file:{self.path.as_posix()}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.path, check_same_thread=False)
            self.conn.executescript(_SCHEMA)
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.row_factory = sqlite3.Row

    # ---- writes -------------------------------------------------------------------------
    def add_documents(self, docs: Iterable[Document]) -> int:
        rows = [
            (
                d.doc_id, d.pmid, d.doi, d.title, d.abstract, d.source_type.value, d.year,
                d.journal, json.dumps(d.pub_types), json.dumps(d.mesh_terms),
                json.dumps(d.authors), d.url, d.source, d.version,
            )  # fmt: skip
            for d in docs
        ]
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
            )
        return len(rows)

    def add_passages(self, passages: Iterable[Passage]) -> int:
        rows = [
            (p.passage_id, p.doc_id, p.ord, p.text, p.char_start, p.char_end, p.n_words)
            for p in passages
        ]
        with self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO passages VALUES (?,?,?,?,?,?,?)", rows)
        return len(rows)

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    # ---- reads --------------------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    @staticmethod
    def _doc(row: sqlite3.Row) -> Document:
        return Document(
            doc_id=row["doc_id"],
            pmid=row["pmid"],
            doi=row["doi"],
            title=row["title"],
            abstract=row["abstract"],
            source_type=SourceType(row["source_type"]),
            year=row["year"],
            journal=row["journal"],
            pub_types=json.loads(row["pub_types"]),
            mesh_terms=json.loads(row["mesh_terms"]),
            authors=json.loads(row["authors"]),
            url=row["url"],
            source=row["source"],
            version=row["version"],
        )

    @staticmethod
    def _passage(row: sqlite3.Row) -> Passage:
        return Passage(**{k: row[k] for k in row.keys()})

    def get_document(self, doc_id: str) -> Document | None:
        row = self.conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return self._doc(row) if row else None

    def get_by_pmid(self, pmid: str) -> Document | None:
        pmid = "".join(ch for ch in str(pmid) if ch.isdigit())
        row = self.conn.execute("SELECT * FROM documents WHERE pmid=?", (pmid,)).fetchone()
        return self._doc(row) if row else None

    def get_by_doi(self, doi: str) -> Document | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE doi=?", (doi.strip().lower(),)
        ).fetchone()
        return self._doc(row) if row else None

    def get_passage(self, passage_id: str) -> Passage | None:
        row = self.conn.execute(
            "SELECT * FROM passages WHERE passage_id=?", (passage_id,)
        ).fetchone()
        return self._passage(row) if row else None

    def get_passages(self, passage_ids: Iterable[str]) -> dict[str, Passage]:
        ids = list(passage_ids)
        out: dict[str, Passage] = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            q = f"SELECT * FROM passages WHERE passage_id IN ({','.join('?' * len(chunk))})"
            for row in self.conn.execute(q, chunk):
                out[row["passage_id"]] = self._passage(row)
        return out

    def passages_of(self, doc_id: str) -> list[Passage]:
        rows = self.conn.execute(
            "SELECT * FROM passages WHERE doc_id=? ORDER BY ord", (doc_id,)
        ).fetchall()
        return [self._passage(r) for r in rows]

    def iter_passages(self, batch: int = 5000) -> Iterator[Passage]:
        cur = self.conn.execute("SELECT * FROM passages ORDER BY rowid")
        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                break
            for r in rows:
                yield self._passage(r)

    def pmids(self) -> set[str]:
        rows = self.conn.execute("SELECT pmid FROM documents WHERE pmid IS NOT NULL")
        return {r[0] for r in rows}

    def counts(self) -> dict[str, int]:
        d = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        p = self.conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        return {"documents": d, "passages": p}

    def close(self) -> None:
        self.conn.close()
