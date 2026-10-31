"""Scope policy: what the system answers, what it escalates.

LitBioRAG supports literature and course-content questions. Patient-specific diagnosis, personal
dosing and emergencies are escalated with a fixed message; they are never answered from
literature. Detection is deliberately conservative regex (false escalations are cheap; a missed
personal-advice question is not). Profiles: ``research`` (default) and ``education`` (adds
exam-style allowances; same escalations).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PATIENT = re.compile(
    r"\b(my|our)\s+(mother|father|mom|dad|son|daughter|wife|husband|child|baby|patient|"
    r"grandmother|grandfather|brother|sister|friend)\b|\bi (have|am having|was diagnosed|take|"
    r"am taking|feel|got)\b|\bshould i (take|stop|start|increase|double)\b|\bfor me\b",
    re.IGNORECASE,
)
_DOSING_PERSONAL = re.compile(
    r"\b(what|which|how much|how many)\s+(dose|dosage|mg|tablets?|pills?)\s+(should|can|do)\s+"
    r"(i|my|we)\b|\bhow (much|many) .{0,40}\bshould i\b",
    re.IGNORECASE,
)
_EMERGENCY = re.compile(
    r"\b(right now|emergency|call 911|unconscious|not breathing|overdos(e|ed|ing)|"
    r"chest pain (now|right now)|suicid(e|al)|poison(ed|ing))\b",
    re.IGNORECASE,
)

ESCALATION_TEXT = {
    "patient_specific": (
        "This system provides information from the biomedical literature and course materials. "
        "It cannot assess an individual patient. Please consult a qualified clinician."
    ),
    "personal_dosing": (
        "Medication doses for a specific person must be decided by a clinician or pharmacist "
        "who knows the patient; this system does not provide them."
    ),
    "emergency": (
        "If this is a medical emergency, contact local emergency services immediately. "
        "This system cannot help with emergencies."
    ),
}


@dataclass(frozen=True)
class ScopeDecision:
    kind: str  # in_scope | patient_specific | personal_dosing | emergency
    matched: str = ""

    @property
    def in_scope(self) -> bool:
        return self.kind == "in_scope"

    @property
    def message(self) -> str:
        return ESCALATION_TEXT.get(self.kind, "")


def classify_scope(question: str, profile: str = "research") -> ScopeDecision:
    q = question or ""
    for kind, pat in (
        ("emergency", _EMERGENCY),
        ("personal_dosing", _DOSING_PERSONAL),
        ("patient_specific", _PATIENT),
    ):
        m = pat.search(q)
        if m:
            return ScopeDecision(kind, m.group(0))
    return ScopeDecision("in_scope")
