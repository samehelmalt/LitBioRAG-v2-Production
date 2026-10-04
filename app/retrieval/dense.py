"""FAISS dense index over passage vectors, built on the build host, memory-mapped when serving.

Layout inside the bundle:  dense/index.faiss, dense/ids.npy (passage_id per row), dense/meta.json.
Small corpora use an exact inner-product index; large ones use IVF-PQ so the index stays on disk
and fits a 16 GB serving host (`faiss.IO_FLAG_MMAP | IO_FLAG_READ_ONLY`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.retrieval.encoders import Encoder, l2_normalize

IVF_THRESHOLD = 50_000  # below this, exact search is cheap and strictly better


@dataclass(frozen=True)
class DenseHit:
    passage_id: str
    score: float


def _choose_index(n: int, dim: int) -> tuple[faiss.Index, dict]:
    if n < IVF_THRESHOLD:
        return faiss.IndexFlatIP(dim), {"type": "FlatIP"}
    nlist = int(min(65_536, max(1024, 4 * math.sqrt(n))))
    m = next(c for c in (64, 48, 32, 16, 8) if dim % c == 0)
    quant = faiss.IndexFlatIP(dim)
    idx = faiss.IndexIVFPQ(quant, dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    return idx, {"type": "IVFPQ", "nlist": nlist, "m": m, "nbits": 8}


def build_dense_index(
    vectors: Iterator[np.ndarray],
    ids: list[str],
    out_dir: Path,
    *,
    encoder_name: str,
    dim: int,
    train_sample: int = 500_000,
) -> dict:
    """Consume vector shards ((k, dim) float32, normalized) in id order; write the index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(ids)
    index, params = _choose_index(n, dim)
    shards = list(vectors)
    if params["type"] == "IVFPQ":
        need = min(n, max(train_sample, 256 * params["nlist"]))
        rng = np.random.default_rng(0)
        pool = np.concatenate(shards)[:n]
        sample = pool[rng.choice(n, size=min(need, n), replace=False)]
        index.train(sample)
        del pool
    added = 0
    for s in shards:
        s = np.ascontiguousarray(s, dtype="float32")
        index.add(s)
        added += len(s)
    if added != n:
        raise ValueError(f"{added} vectors but {n} ids")
    faiss.write_index(index, str(out_dir / "index.faiss"))
    np.save(out_dir / "ids.npy", np.asarray(ids, dtype=object), allow_pickle=True)
    meta = {"encoder": encoder_name, "dim": dim, "n": n, **params}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


class DenseIndex:
    def __init__(self, path: Path | str, encoder: Encoder, *, mmap: bool = True, nprobe: int = 64):
        self.path = Path(path)
        self.encoder = encoder
        self.meta = json.loads((self.path / "meta.json").read_text())
        if self.meta["encoder"] != encoder.name:
            raise ValueError(
                f"index built with {self.meta['encoder']!r}, got encoder {encoder.name!r}"
            )
        flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY if mmap else 0
        try:
            self.index = faiss.read_index(str(self.path / "index.faiss"), flags)
        except RuntimeError:
            self.index = faiss.read_index(str(self.path / "index.faiss"))
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe
        self.ids = np.load(self.path / "ids.npy", allow_pickle=True)

    def search(self, query: str, k: int = 100) -> list[DenseHit]:
        return self.search_many([query], k)[0]

    def search_many(self, queries: list[str], k: int = 100) -> list[list[DenseHit]]:
        q = l2_normalize(self.encoder.encode(queries))
        scores, idxs = self.index.search(np.ascontiguousarray(q, dtype="float32"), k)
        out: list[list[DenseHit]] = []
        for row_s, row_i in zip(scores, idxs, strict=True):
            hits = [
                DenseHit(str(self.ids[i]), float(s))
                for s, i in zip(row_s, row_i, strict=True)
                if 0 <= i < len(self.ids)
            ]
            out.append(hits)
        return out
