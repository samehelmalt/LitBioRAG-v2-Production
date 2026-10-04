"""Trust metrics: gate decisions against labels, calibration, coverage-risk.

Labels are booleans: ``True`` means the answer was acceptable to release (human label when
available, otherwise the proxy defined in evaluation/labels.py). Decisions are PASS/WARN/BLOCK.

Definitions
* false PASS  = PASS given to an unacceptable answer           (primary safety metric)
* false WARN  = WARN given to an unacceptable answer (released with a caveat)
* false BLOCK = BLOCK given to an acceptable answer            (lost coverage)
* abstention precision = share of BLOCKs that were on unacceptable answers
* abstention recall    = share of unacceptable answers that were BLOCKed
* coverage = share of answers released (PASS or WARN)
* ECE / Brier are computed on a confidence in [0, 1] against the label.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

DECISIONS = ("PASS", "WARN", "BLOCK")


@dataclass(frozen=True)
class GateConfusion:
    n: int
    counts: dict  # (decision, label) -> count
    false_pass_rate: float | None  # among PASS decisions
    false_warn_rate: float | None  # among WARN decisions
    false_block_rate: float | None  # among BLOCK decisions
    unacceptable_released_rate: float | None  # among unacceptable answers: PASS or WARN
    abstention_precision: float | None
    abstention_recall: float | None
    coverage: float
    pass_rate: float
    warn_rate: float
    block_rate: float


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def gate_confusion(decisions: Sequence[str], acceptable: Sequence[bool]) -> GateConfusion:
    if len(decisions) != len(acceptable):
        raise ValueError("decisions and labels differ in length")
    n = len(decisions)
    if n == 0:
        raise ValueError("no rows")
    counts: dict[tuple[str, bool], int] = {}
    for d, a in zip(decisions, acceptable, strict=True):
        if d not in DECISIONS:
            raise ValueError(f"unknown decision {d!r}")
        counts[(d, bool(a))] = counts.get((d, bool(a)), 0) + 1

    def c(d: str, a: bool) -> int:
        return counts.get((d, a), 0)

    n_pass = c("PASS", True) + c("PASS", False)
    n_warn = c("WARN", True) + c("WARN", False)
    n_block = c("BLOCK", True) + c("BLOCK", False)
    n_bad = sum(1 for a in acceptable if not a)
    return GateConfusion(
        n=n,
        counts={f"{d}|{'ok' if a else 'bad'}": v for (d, a), v in sorted(counts.items())},
        false_pass_rate=_ratio(c("PASS", False), n_pass),
        false_warn_rate=_ratio(c("WARN", False), n_warn),
        false_block_rate=_ratio(c("BLOCK", True), n_block),
        unacceptable_released_rate=_ratio(c("PASS", False) + c("WARN", False), n_bad),
        abstention_precision=_ratio(c("BLOCK", False), n_block),
        abstention_recall=_ratio(c("BLOCK", False), n_bad),
        coverage=(n_pass + n_warn) / n,
        pass_rate=n_pass / n,
        warn_rate=n_warn / n,
        block_rate=n_block / n,
    )


def brier(confidence: Sequence[float], acceptable: Sequence[bool]) -> float:
    if len(confidence) != len(acceptable) or not confidence:
        raise ValueError("inputs must be non-empty and equal length")
    return sum((p - float(a)) ** 2 for p, a in zip(confidence, acceptable, strict=True)) / len(
        confidence
    )


def ece(confidence: Sequence[float], acceptable: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    if len(confidence) != len(acceptable) or not confidence:
        raise ValueError("inputs must be non-empty and equal length")
    n = len(confidence)
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for p, a in zip(confidence, acceptable, strict=True):
        if not 0.0 <= p <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        idx = min(int(p * bins), bins - 1)
        buckets[idx].append((p, bool(a)))
    total = 0.0
    for b in buckets:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(a for _, a in b) / len(b)
        total += len(b) / n * abs(acc - conf)
    return total


def coverage_risk_curve(
    confidence: Sequence[float], acceptable: Sequence[bool]
) -> list[tuple[float, float, float]]:
    """Sort by confidence descending; at each cut return (threshold, coverage, risk).

    risk = share of released answers that are unacceptable. The curve answers: if we release
    only answers with confidence >= t, how much do we release and how often are we wrong?
    """
    if len(confidence) != len(acceptable) or not confidence:
        raise ValueError("inputs must be non-empty and equal length")
    order = sorted(zip(confidence, acceptable, strict=True), key=lambda t: -t[0])
    n = len(order)
    out: list[tuple[float, float, float]] = []
    bad = 0
    for i, (p, a) in enumerate(order, start=1):
        bad += 0 if a else 1
        if i == n or order[i][0] != p:  # emit once per distinct threshold
            out.append((p, i / n, bad / i))
    return out


def risk_at_coverage(curve: Sequence[tuple[float, float, float]], coverage: float) -> float | None:
    """Smallest-coverage point on the curve with coverage >= requested; None if unreachable."""
    for _, cov, risk in curve:
        if cov >= coverage:
            return risk
    return None
