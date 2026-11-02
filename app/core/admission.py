"""Admission control: bounded concurrency per model role with a bounded wait queue.

One process serves many students at once. An in-process generator (``hf_causal``) handles one
sequence at a time, so ten simultaneous questions mean contention at best and a CUDA out-of-memory
at worst. A verifier or reranker can take a few requests at once but not fifty. ``Admission``
gives every role a capacity (how many requests may hold the model at once) and a queue bound (how
many may wait). A caller that would exceed the queue bound, or waits longer than the timeout, is
refused with :class:`Overloaded`, which the API turns into ``503`` + ``Retry-After``: the service
degrades to "try again shortly" instead of crashing or stalling everyone.

The queue is FIFO; a waiting caller learns its position so the progress stream can show
"queued (3 ahead)". Counters (in flight, queued, admitted, rejected, timed out, wait time) feed
``/health``. No model is touched here; the pool wraps its ``use()`` with ``admit()``.

Shared state is per process: with several API replicas each one admits independently, which is
the intended shape (each replica owns its GPU or its share of a vLLM pool).
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


class Overloaded(RuntimeError):
    def __init__(self, role: str, reason: str, retry_after_s: float):
        super().__init__(f"{role}: {reason}")
        self.role, self.reason, self.retry_after_s = role, reason, retry_after_s


@dataclass
class RoleLimits:
    capacity: int = 1  # concurrent holders
    queue_max: int = 16  # waiters beyond capacity; 0 = refuse immediately when busy
    queue_timeout_s: float = 30.0  # longest wait before Overloaded


@dataclass
class _RoleState:
    limits: RoleLimits
    in_flight: int = 0
    waiting: deque = field(default_factory=deque)  # tickets in FIFO order
    admitted: int = 0
    rejected: int = 0
    timed_out: int = 0
    wait_ms_total: float = 0.0
    max_wait_ms: float = 0.0


class Admission:
    """Per-role gates. ``limits`` maps role -> RoleLimits; unknown roles use ``default`` or are
    unlimited when ``default`` is None."""

    def __init__(self, limits: dict[str, RoleLimits] | None = None,
                 default: RoleLimits | None = None):  # fmt: skip
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._default = default
        self._roles: dict[str, _RoleState] = {
            r: _RoleState(lim) for r, lim in (limits or {}).items()
        }

    @classmethod
    def from_config(cls, raw: dict[str, Any] | None) -> Admission:
        """``configs/resources.yaml: concurrency`` →

            concurrency:
              queue_max: 32              # defaults for every role
              queue_timeout_s: 30
              roles:
                generator: {capacity: 1}
                verifier: {capacity: 4, queue_max: 64}
        """
        raw = raw or {}
        env = {k[len("AEOS_CONCURRENCY_"):].lower(): v for k, v in os.environ.items()
               if k.startswith("AEOS_CONCURRENCY_") and v.strip()}  # AEOS_CONCURRENCY_GENERATOR=32
        base = RoleLimits(
            capacity=int(raw.get("capacity", 1)),
            queue_max=int(raw.get("queue_max", 16)),
            queue_timeout_s=float(raw.get("queue_timeout_s", 30.0)),
        )
        limits: dict[str, RoleLimits] = {}
        roles = dict(raw.get("roles") or {})
        for role in env:
            roles.setdefault(role, {})
        for role, spec in roles.items():
            spec = dict(spec or {})
            if role in env:
                spec["capacity"] = int(env[role])  # deployment override (e.g. vLLM batch size)
            limits[role] = RoleLimits(
                capacity=int(spec.get("capacity", base.capacity)),
                queue_max=int(spec.get("queue_max", base.queue_max)),
                queue_timeout_s=float(spec.get("queue_timeout_s", base.queue_timeout_s)),
            )
        return cls(limits, default=None)  # roles not listed are not gated

    def limits(self, role: str) -> RoleLimits | None:
        st = self._roles.get(role)
        if st is not None:
            return st.limits
        return self._default

    def _state(self, role: str) -> _RoleState | None:
        st = self._roles.get(role)
        if st is None and self._default is not None:
            st = self._roles[role] = _RoleState(self._default)
        return st

    @contextmanager
    def admit(self, role: str, *, on_queued: Callable[[int], None] | None = None,
              timeout_s: float | None = None) -> Iterator[None]:  # fmt: skip
        """Hold one slot of ``role`` for the block. Waits in FIFO order while the role is at
        capacity; ``on_queued(position)`` is called once with the number of callers ahead."""
        with self._lock:
            st = self._state(role)
            if st is None:
                gated = False
            else:
                gated = True
                self._acquire(role, st, on_queued, timeout_s)
        try:
            yield
        finally:
            if gated:
                with self._lock:
                    st.in_flight -= 1
                    self._cv.notify_all()

    def _acquire(self, role: str, st: _RoleState, on_queued, timeout_s) -> None:
        lim = st.limits
        if st.in_flight < lim.capacity and not st.waiting:
            st.in_flight += 1
            st.admitted += 1
            return
        if len(st.waiting) >= lim.queue_max:
            st.rejected += 1
            raise Overloaded(role, f"{len(st.waiting)} requests already waiting",
                             self._retry_after(st))  # fmt: skip
        ticket = object()
        position = len(st.waiting)
        st.waiting.append(ticket)
        if on_queued is not None:
            try:
                on_queued(position)
            except Exception:  # noqa: BLE001  (progress callbacks never break admission)
                pass
        deadline = time.monotonic() + (lim.queue_timeout_s if timeout_s is None else timeout_s)
        t0 = time.monotonic()
        try:
            while not (st.waiting[0] is ticket and st.in_flight < lim.capacity):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    st.timed_out += 1
                    raise Overloaded(role, "waited too long for a free slot", self._retry_after(st))
                self._cv.wait(remaining)
        finally:
            st.waiting.remove(ticket)
            self._cv.notify_all()  # the next ticket may now be at the head
        waited = (time.monotonic() - t0) * 1000
        st.wait_ms_total += waited
        st.max_wait_ms = max(st.max_wait_ms, waited)
        st.in_flight += 1
        st.admitted += 1

    @staticmethod
    def _retry_after(st: _RoleState) -> float:
        """A rough hint: average wait so far, at least a few seconds."""
        waits = max(1, st.admitted)
        avg_s = st.wait_ms_total / waits / 1000.0
        return max(5.0, min(120.0, round(avg_s + 5.0)))

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            out: dict[str, dict[str, Any]] = {}
            for role, st in self._roles.items():
                admitted = max(1, st.admitted)
                out[role] = {
                    "capacity": st.limits.capacity, "queue_max": st.limits.queue_max,
                    "in_flight": st.in_flight, "queued": len(st.waiting),
                    "admitted": st.admitted, "rejected": st.rejected, "timed_out": st.timed_out,
                    "avg_wait_ms": round(st.wait_ms_total / admitted, 1),
                    "max_wait_ms": round(st.max_wait_ms, 1),
                }  # fmt: skip
            return out
