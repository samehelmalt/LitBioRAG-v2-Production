"""Incremental dense index for uploaded knowledge bases.

``faiss.IndexIDMap2(IndexFlatIP)`` keeps an int64 id per vector, so passages can be added and
removed without re-embedding everything (the previous implementation rebuilt the whole index on
every change). Persistence is atomic: the index and the id map are written to temp files and
renamed into place, so a concurrent reader never sees a half-written index. A full rebuild is
needed only when the encoder changes (the stored encoder name is checked on load).
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np

from app.retrieval.dense import DenseHit
from app.retrieval.encoders import Encoder, encode_queries, l2_normalize


class IncrementalDense:
    def __init__(self, path: Path | str, encoder: Encoder):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder
        self._lock = threading.RLock()
        self._next_id = 0
        self._ids: dict[int, str] = {}
        self._rev: dict[str, int] = {}
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(encoder.dim))
        self._load()

    # ---- persistence ------------------------------------------------------------------------
    def _load(self) -> None:
        meta_p = self.path / "meta.json"
        if not meta_p.exists():
            return
        meta = json.loads(meta_p.read_text())
        if meta.get("encoder") != self.encoder.name or meta.get("dim") != self.encoder.dim:
            return  # encoder changed: start empty; the indexer re-embeds from the store
        self.index = faiss.read_index(str(self.path / "index.faiss"))
        ids = json.loads((self.path / "ids.json").read_text())
        self._ids = {int(k): v for k, v in ids.items()}
        self._rev = {v: k for k, v in self._ids.items()}
        self._next_id = (max(self._ids) + 1) if self._ids else 0

    def _save(self) -> None:
        tmp_idx = self.path / "index.faiss.tmp"
        tmp_ids = self.path / "ids.json.tmp"
        tmp_meta = self.path / "meta.json.tmp"
        faiss.write_index(self.index, str(tmp_idx))
        tmp_ids.write_text(json.dumps(self._ids))
        tmp_meta.write_text(json.dumps({"encoder": self.encoder.name, "dim": self.encoder.dim,
                                        "n": self.index.ntotal, "type": "IDMap2-FlatIP"}))
        os.replace(tmp_idx, self.path / "index.faiss")
        os.replace(tmp_ids, self.path / "ids.json")
        os.replace(tmp_meta, self.path / "meta.json")

    # ---- mutation ---------------------------------------------------------------------------
    def add(self, passage_ids: Sequence[str], texts: Sequence[str], *, batch_size: int = 64) -> int:
        if not passage_ids:
            return 0
        with self._lock:
            self.remove([p for p in passage_ids if p in self._rev], save=False)
            vecs = np.ascontiguousarray(
                l2_normalize(self.encoder.encode(list(texts), batch_size=batch_size)),
                dtype="float32")
            ids = np.arange(self._next_id, self._next_id + len(passage_ids), dtype="int64")
            self.index.add_with_ids(vecs, ids)
            for i, pid in zip(ids.tolist(), passage_ids, strict=True):
                self._ids[i] = pid
                self._rev[pid] = i
            self._next_id += len(passage_ids)
            self._save()
        return len(passage_ids)

    def remove(self, passage_ids: Sequence[str], *, save: bool = True) -> int:
        with self._lock:
            ids = [self._rev[p] for p in passage_ids if p in self._rev]
            if ids:
                self.index.remove_ids(np.asarray(ids, dtype="int64"))
                for i in ids:
                    self._rev.pop(self._ids.pop(i), None)
            if save and ids:
                self._save()
            return len(ids)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    # ---- search (same contract as app.retrieval.dense.DenseIndex) ---------------------------
    def search(self, query: str, k: int = 100) -> list[DenseHit]:
        return self.search_many([query], k)[0]

    def search_many(self, queries: list[str], k: int = 100) -> list[list[DenseHit]]:
        with self._lock:
            if self.index.ntotal == 0:
                return [[] for _ in queries]
            q = l2_normalize(encode_queries(self.encoder, queries))
            k = min(k, self.index.ntotal)
            scores, idxs = self.index.search(np.ascontiguousarray(q, dtype="float32"), k)
            out: list[list[DenseHit]] = []
            for row_s, row_i in zip(scores, idxs, strict=True):
                out.append([DenseHit(self._ids[int(i)], float(s))
                            for s, i in zip(row_s, row_i, strict=True)
                            if int(i) in self._ids])  # fmt: skip
            return out
