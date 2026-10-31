"""Small on-disk cache (SQLite) for expensive, deterministic model calls.

Keys always include the model name and revision, so swapping a model in models.yaml never serves
stale results. Used for verifier (premise, hypothesis) scores and query embeddings; generations are
not cached by default (configs/resources.yaml) because the evidence set changes with the bundle.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


class DiskCache:
    def __init__(self, path: Path | str, *, ttl_days: float = 30.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_s = ttl_days * 86400
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (ns TEXT, k TEXT, v TEXT, ts REAL, PRIMARY KEY (ns, k))"
        )
        self.hits = 0
        self.misses = 0

    def get(self, ns: str, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT v, ts FROM kv WHERE ns = ? AND k = ?", (ns, key)
            ).fetchone()
        if row is None or (self.ttl_s and time.time() - row[1] > self.ttl_s):
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, ns: str, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (ns, k, v, ts) VALUES (?, ?, ?, ?)",
                (ns, key, json.dumps(value), time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class CachedVerifier:
    """Wraps any ``Verifier``; scores only the pairs the cache does not already hold."""

    def __init__(self, inner, cache: DiskCache):
        self.inner = inner
        self.cache = cache
        self.name = inner.name

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list:
        from app.verification.base import NLIScores

        out: list[NLIScores | None] = [None] * len(pairs)
        todo: list[int] = []
        for i, (p, h) in enumerate(pairs):
            hit = self.cache.get("nli", _key(self.name, p, h))
            if hit is None:
                todo.append(i)
            else:
                out[i] = NLIScores.model_validate(hit)
        if todo:
            fresh = self.inner.score_pairs([pairs[i] for i in todo])
            for i, s in zip(todo, fresh, strict=True):
                out[i] = s
                self.cache.put("nli", _key(self.name, *pairs[i]), s.model_dump())
        return out  # type: ignore[return-value]


class CachedEncoder:
    """Wraps an ``Encoder``; caches per-text vectors (queries are short and repeat often)."""

    def __init__(self, inner, cache: DiskCache):
        self.inner = inner
        self.cache = cache
        self.name = inner.name
        self.dim = inner.dim

    def encode(self, texts: Sequence[str], batch_size: int = 64):
        import numpy as np

        out = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            hit = self.cache.get("emb", _key(self.name, t))
            if hit is None:
                todo.append(i)
            else:
                out[i] = np.asarray(hit, dtype="float32")
        if todo:
            fresh = self.inner.encode([texts[i] for i in todo], batch_size=batch_size)
            for i, v in zip(todo, fresh, strict=True):
                out[i] = np.asarray(v, dtype="float32")
                self.cache.put("emb", _key(self.name, texts[i]), out[i].tolist())
        return np.stack(out)
