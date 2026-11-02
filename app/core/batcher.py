"""Cross-request micro-batching for pair-scoring models (verifier NLI, cross-encoder reranker).

Each request scores its own (premise, hypothesis) or (query, passage) pairs in one batched forward
pass. With many students asking at once, the GPU still runs one forward pass per request: ten
requests, ten passes, each padded to its own longest pair. ``MicroBatcher`` lets concurrent callers
share passes: the first caller of a window waits up to ``max_wait_ms`` for others, then one call of
``fn`` scores every pending item and each caller gets its own slice back, in order.

Uncontended, there is no wait: a caller that finds nobody else inside ``submit`` runs ``fn``
directly, so a single user pays nothing for the mechanism. The batcher never changes results
(``fn`` must be order-preserving and item-wise), only when the forward pass happens; the tests
compare batched against direct scores exactly.

Wrappers: ``BatchedVerifier`` (any ``Verifier``) and ``BatchedReranker`` (any reranker that
exposes ``score_pairs(query, text) -> float``; others are returned unchanged). The registry wraps
roles listed in ``configs/resources.yaml: batching.roles``; the on-disk cache, when enabled, sits
outside the batcher so cache hits never wait.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class _Request(Generic[T, R]):
    __slots__ = ("items", "done", "result", "error")

    def __init__(self, items: Sequence[T]):
        self.items = list(items)
        self.done = threading.Event()
        self.result: list[R] | None = None
        self.error: BaseException | None = None


class MicroBatcher(Generic[T, R]):
    def __init__(self, fn: Callable[[list[T]], list[R]], *, max_wait_ms: float = 20.0,
                 max_batch: int = 64, name: str = ""):  # fmt: skip
        if max_batch < 1:
            raise ValueError("max_batch must be positive")
        self.fn = fn
        self.max_wait_s = max(0.0, max_wait_ms) / 1000.0
        self.max_batch = max_batch
        self.name = name
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pending: list[_Request[T, R]] = []
        self._leader: _Request[T, R] | None = None
        self._inside = 0  # callers currently inside submit()
        # metrics
        self.direct_calls = 0
        self.batches = 0
        self.batched_items = 0
        self.max_batch_seen = 0

    # ---- public -------------------------------------------------------------------------------
    def submit(self, items: Sequence[T]) -> list[R]:
        if not items:
            return []
        with self._lock:
            self._inside += 1
            alone = self._inside == 1 and not self._pending
        try:
            if alone:
                with self._lock:
                    self.direct_calls += 1
                return list(self.fn(list(items)))
            return self._batched(items)
        finally:
            with self._lock:
                self._inside -= 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            b = max(1, self.batches)
            return {"direct_calls": self.direct_calls, "batches": self.batches,
                    "batched_items": self.batched_items,
                    "avg_batch": round(self.batched_items / b, 1),
                    "max_batch": self.max_batch_seen}  # fmt: skip

    # ---- internals ----------------------------------------------------------------------------
    def _batched(self, items: Sequence[T]) -> list[R]:
        req: _Request[T, R] = _Request(items)
        with self._lock:
            self._pending.append(req)
            lead = self._leader is None
            if lead:
                self._leader = req
            self._cv.notify_all()
        if lead:
            self._lead()
        else:
            req.done.wait()
        if req.error is not None:
            raise req.error
        return req.result  # type: ignore[return-value]

    def _lead(self) -> None:
        deadline = time.monotonic() + self.max_wait_s
        with self._lock:
            while True:
                n = sum(len(r.items) for r in self._pending)
                remaining = deadline - time.monotonic()
                if n >= self.max_batch or remaining <= 0:
                    break
                self._cv.wait(remaining)
            batch, self._pending, self._leader = self._pending, [], None
        flat: list[T] = [it for r in batch for it in r.items]
        try:
            out = list(self.fn(flat))
            if len(out) != len(flat):
                raise RuntimeError(f"{self.name or 'batcher'}: fn returned {len(out)} results "
                                   f"for {len(flat)} items")  # fmt: skip
            pos = 0
            for r in batch:
                r.result = out[pos: pos + len(r.items)]
                pos += len(r.items)
        except BaseException as e:  # noqa: BLE001  (every waiter must learn about the failure)
            for r in batch:
                r.error = e
        finally:
            with self._lock:
                self.batches += 1
                self.batched_items += len(flat)
                self.max_batch_seen = max(self.max_batch_seen, len(flat))
            for r in batch:
                r.done.set()


# ---------------------------------------------------------------------------------- wrappers ----
class BatchedVerifier:
    """Any ``Verifier``; concurrent ``score_pairs`` calls share forward passes."""

    def __init__(self, inner, *, max_wait_ms: float = 20.0, max_batch: int = 64):
        self.inner = inner
        self.name = inner.name
        self.batcher: MicroBatcher = MicroBatcher(inner.score_pairs, max_wait_ms=max_wait_ms,
                                                  max_batch=max_batch, name=inner.name)  # fmt: skip

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list:
        return self.batcher.submit(pairs)

    def stats(self) -> dict[str, Any]:
        return self.batcher.stats()


class BatchedReranker:
    """A reranker with ``score_pairs([(query, text)]) -> [float]``; rerank() builds the pairs,
    submits them to the shared batcher, and sorts its own slice."""

    def __init__(self, inner, *, max_wait_ms: float = 20.0, max_batch: int = 64):
        if not hasattr(inner, "score_pairs"):
            raise TypeError(f"{inner.name} cannot be batched: no score_pairs()")
        self.inner = inner
        self.name = inner.name
        self.batcher: MicroBatcher = MicroBatcher(inner.score_pairs, max_wait_ms=max_wait_ms,
                                                  max_batch=max_batch, name=inner.name)  # fmt: skip

    def rerank(self, query: str, candidates: Sequence, top_n: int) -> list:
        from app.reranking.base import RerankHit

        if not candidates:
            return []
        scores = self.batcher.submit([(query, c.text) for c in candidates])
        hits = [RerankHit(c.passage_id, float(s)) for c, s in zip(candidates, scores, strict=True)]
        hits.sort(key=lambda h: (-h.score, h.passage_id))
        return hits[:top_n]

    def stats(self) -> dict[str, Any]:
        return self.batcher.stats()


def wrap_with_batching(role: str, obj: Any, cfg: dict[str, Any] | None) -> Any:
    """Apply ``resources.yaml: batching`` to a built model; returns ``obj`` when not applicable."""
    cfg = cfg or {}
    if not cfg.get("enabled") or role not in (cfg.get("roles") or ["verifier", "reranker"]):
        return obj
    kw = {"max_wait_ms": float(cfg.get("max_wait_ms", 20)),
          "max_batch": int(cfg.get("max_batch", 64))}  # fmt: skip
    if role == "verifier" and hasattr(obj, "score_pairs"):
        return BatchedVerifier(obj, **kw)
    if role == "reranker" and hasattr(obj, "score_pairs"):
        return BatchedReranker(obj, **kw)
    return obj
