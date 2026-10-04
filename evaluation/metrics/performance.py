"""Latency and resource summaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        raise ValueError("no values")
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@dataclass(frozen=True)
class LatencySummary:
    n: int
    p50: float
    p95: float
    p99: float
    mean: float
    max: float


def latency_summary(seconds: Sequence[float]) -> LatencySummary:
    return LatencySummary(
        n=len(seconds),
        p50=percentile(seconds, 50),
        p95=percentile(seconds, 95),
        p99=percentile(seconds, 99),
        mean=sum(seconds) / len(seconds),
        max=max(seconds),
    )
