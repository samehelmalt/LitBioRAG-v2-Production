"""Per-principal token bucket. In-process; a multi-replica deployment puts this in Redis."""

from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, rate_per_minute: int, burst: int):
        self.rate = rate_per_minute / 60.0
        self.burst = max(1, burst)
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                return True, 0.0
            self._buckets[key] = (tokens, now)
            return False, (1.0 - tokens) / self.rate if self.rate else 60.0
