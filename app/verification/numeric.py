"""Numeric consistency: every number in a claim must have a counterpart in the supporting evidence
with a compatible unit class and within tolerance.

The thesis checked substring presence of the digits anywhere in the context (so "23%" matched
"123%" and a number attached to the wrong group passed). Here each mention is parsed with its unit
class (percent, p-value, ratio, mass, volume, time, count, plain) and compared only against
mentions of the same class. Tuple-level role checking (which group, which outcome) is the next
refinement and is recorded as a limitation in TRUST_AND_SAFETY.md.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

_NUM = r"(?P<num>[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)"
_UNITS = {
    "percent": r"%|percent|per\s*cent",
    "p_value": None,  # handled by its own pattern
    "ratio": r"(?:HR|OR|RR|IRR|aHR|aOR)\b",
    "mass": r"(?:mg|µg|ug|mcg|g|kg|ng|pg)(?:/(?:kg|day|d|dl|dL|l|L|m2|mL|ml))?\b",
    "volume": r"(?:ml|mL|l|L|dl|dL|µl|ul)\b",
    "time": r"(?:years?|yrs?|months?|mo|weeks?|wks?|days?|d|hours?|hrs?|h|minutes?|min)\b",
    "count": r"(?:patients?|participants?|subjects?|cases?|controls?|women|men|children|adults|"
             r"individuals|samples?)\b",  # fmt: skip
}
_P_RE = re.compile(r"\bp\s*(?:=|<|>|≤|≥)\s*(?P<num>0?\.\d+|\d(?:\.\d+)?e-?\d+)", re.I)
_RATIO_PREFIX_RE = re.compile(
    r"\b(?:HR|OR|RR|IRR|aHR|aOR)\s*(?:=|of|:)?\s*" + _NUM, re.I
)
_GENERIC_RE = re.compile(
    _NUM
    + r"\s*(?P<unit>"
    + "|".join(p for p in _UNITS.values() if p)
    + r")?",
    re.I,
)
_CI_RE = re.compile(r"95\s*%\s*CI", re.I)


@dataclass(frozen=True)
class NumberMention:
    value: float
    unit_class: str  # percent | p_value | ratio | mass | volume | time | count | plain
    raw: str


def _unit_class(unit: str | None) -> str:
    if not unit:
        return "plain"
    u = unit.lower()
    for cls, pat in _UNITS.items():
        if pat and re.fullmatch(pat, unit, re.I):
            return cls
    if u.startswith("per") or u == "%":
        return "percent"
    return "plain"


def extract_numbers(text: str) -> list[NumberMention]:
    text = _CI_RE.sub("CI", text)  # the '95' in '95% CI' is not a finding
    out: list[NumberMention] = []
    taken: list[tuple[int, int]] = []
    for m in _P_RE.finditer(text):
        out.append(NumberMention(float(m.group("num")), "p_value", m.group(0)))
        taken.append(m.span())
    for m in _RATIO_PREFIX_RE.finditer(text):
        out.append(NumberMention(float(m.group("num").replace(",", "")), "ratio", m.group(0)))
        taken.append(m.span())
    for m in _GENERIC_RE.finditer(text):
        if any(a <= m.start() < b for a, b in taken):
            continue
        raw_num = m.group("num").replace(",", "")
        try:
            val = float(raw_num)
        except ValueError:
            continue
        out.append(NumberMention(val, _unit_class(m.group("unit")), m.group(0).strip()))
    return out


def _match(a: NumberMention, b: NumberMention, rel_tol: float, abs_tol: float) -> bool:
    if a.unit_class != b.unit_class and not (
        {a.unit_class, b.unit_class} == {"plain", "percent"} and a.value == b.value
    ):
        return False
    return math.isclose(a.value, b.value, rel_tol=rel_tol, abs_tol=abs_tol)


def numeric_consistency(
    claim_text: str,
    supporting_texts: Iterable[str],
    *,
    rel_tol: float = 0.02,
    abs_tol: float = 0.5,
) -> tuple[int, int, tuple[str, ...]]:
    """Return (checked, ok, mismatches) for the numbers in the claim."""
    claim_nums = extract_numbers(claim_text)
    if not claim_nums:
        return 0, 0, ()
    pool: list[NumberMention] = []
    for t in supporting_texts:
        pool.extend(extract_numbers(t))
    mismatches = tuple(
        n.raw for n in claim_nums if not any(_match(n, e, rel_tol, abs_tol) for e in pool)
    )
    return len(claim_nums), len(claim_nums) - len(mismatches), mismatches
