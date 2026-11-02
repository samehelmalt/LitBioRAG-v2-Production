"""Follow-up resolution from the approved conversation state.

"Why is that contraindicated?" cannot be retrieved as written. The resolver rewrites such
questions into a standalone form using only the previous *delivered* turn (question + answer
entities), shows the user what it understood ("Interpreted as: …"), and retrieval runs on the
rewritten question. Earlier answers remain context, never evidence. Rule-based by design: no
model call, no new facts; an optional generator rewrite can be layered on later with the same
contract (``Resolution``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PRONOUNS = re.compile(r"\b(that|this|it|its|those|these|them|the other|the same|the next|the "
                       r"first|the second|the latter|the former)\b", re.I)
_ELLIPSIS = re.compile(r"^\s*(and|so|then|also|what about|how about|why|how fast|how much|how "
                       r"many|which one|what else)\b", re.I)
_HISTORY_LINE = re.compile(r"^Q:\s*(?P<q>.+?)\s*→\s*delivered answer:\s*(?P<a>.*)$", re.S)
_NO_ANSWER = re.compile(r"\(no answer was delivered\)")


@dataclass(frozen=True)
class Resolution:
    question: str  # what retrieval and generation use
    interpreted: bool  # True when the question was rewritten
    based_on: str = ""  # the previous question used
    entities: tuple[str, ...] = ()
    note: str = ""


def _last_delivered(history: list[str]) -> tuple[str, str] | None:
    for line in reversed(history or []):
        if _NO_ANSWER.search(line):
            continue
        m = _HISTORY_LINE.match(line)
        if m:
            return m.group("q").strip(), m.group("a").strip()
    return None


def is_follow_up(question: str) -> bool:
    q = question.strip()
    short = len(q.split()) <= 12
    return bool(_ELLIPSIS.match(q)) or (short and bool(_PRONOUNS.search(q)))


def resolve(question: str, history: list[str], extract_entities=None) -> Resolution:
    """Rewrite a follow-up using the last delivered turn; otherwise return it unchanged."""
    q = question.strip()
    if not history or not is_follow_up(q):
        return Resolution(q, False)
    prev = _last_delivered(history)
    if prev is None:
        return Resolution(q, False, note="no delivered answer to resolve against")
    prev_q, prev_a = prev
    ents: list[str] = []
    if extract_entities is not None:
        try:
            seen = set()
            for src in (prev_q, prev_a):
                for e in extract_entities(src):
                    if e not in seen:
                        seen.add(e)
                        ents.append(e)
        except Exception:  # noqa: BLE001
            ents = []
    topic = ", ".join(ents[:4]) if ents else prev_q
    resolved = f"{q.rstrip('?')} — regarding: {topic}? (follow-up to: {prev_q})"
    return Resolution(resolved, True, based_on=prev_q, entities=tuple(ents[:8]),
                      note="rewritten from the previous delivered answer; earlier answers are "
                           "context, not evidence")  # fmt: skip
