"""Output-integrity checks on a generation attempt, before verification.

These detect the defects seen in the LitBioRAG experiments — repeated sentences and n-gram loops
(DeepSeek), role echoes ("User:" / "Assistant:"), truncated JSON, duplicated claims, answer text
not covered by claims, and answers that ignore the question's entities. A defect **rejects the
attempt**; nothing here edits the text. Clinically meaningful repetition is not silently removed:
the attempt is recorded as defective and the generator is asked again with precise feedback
(``app/generation/attempts.py``), or the system abstains.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from app.generation.contract import GeneratedAnswer
from app.retrieval.passages import split_sentences

_ROLE_ECHO = re.compile(r"(?im)^\s*(user|assistant|system|human|ai)\s*:")
_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class QualityReport:
    ok: bool = True
    defects: list[str] = field(default_factory=list)
    repetition_ratio: float = 0.0
    ngram_loop_ratio: float = 0.0
    duplicate_claims: int = 0
    uncovered_sentences: int = 0
    role_echo: bool = False
    truncated_json: bool = False
    off_topic: bool = False

    def add(self, defect: str) -> None:
        self.ok = False
        self.defects.append(defect)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _norm(s: str) -> str:
    return " ".join(_WORD.findall(s.lower()))


def repetition_ratio(text: str) -> float:
    """Share of sentences that repeat an earlier sentence (normalized)."""
    spans = split_sentences(text)
    sents = [_norm(text[s:e]) for s, e in spans]
    sents = [s for s in sents if len(s) >= 12]
    if len(sents) < 2:
        return 0.0
    seen: set[str] = set()
    rep = 0
    for s in sents:
        if s in seen:
            rep += 1
        seen.add(s)
    return rep / len(sents)


def ngram_loop_ratio(text: str, n: int = 6) -> float:
    """Share of n-grams that occur more than once (a loop makes this climb fast)."""
    words = _WORD.findall(text.lower())
    if len(words) < n * 3:
        return 0.0
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    repeated = sum(c - 1 for c in grams.values() if c > 1)
    return repeated / max(1, len(words) - n + 1)


def looks_truncated(raw: str) -> bool:
    s = raw.strip()
    if not s:
        return True
    opens, closes = s.count("{"), s.count("}")
    return opens > closes or s.endswith(("{", "[", ",", ":")) or (opens and not s.endswith("}"))


def assess(
    raw: str,
    parsed: GeneratedAnswer,
    *,
    question_entities: set[str] | None = None,
    claim_entities: set[str] | None = None,
    max_repetition: float = 0.2,
    max_ngram_loop: float = 0.15,
    max_uncovered: int = 2,
) -> QualityReport:
    rep = QualityReport()
    if parsed.abstained:
        if not parsed.format_ok:
            rep.truncated_json = looks_truncated(raw)
            rep.add("unparseable output" + (" (truncated)" if rep.truncated_json else ""))
        return rep  # a clean abstention is a valid outcome
    # repetition is judged on the answer text (claims restate it by design); duplicate claims
    # are counted separately below
    text_all = parsed.answer or " ".join(c.text for c in parsed.claims)
    rep.repetition_ratio = repetition_ratio(text_all)
    if rep.repetition_ratio > max_repetition:
        rep.add(f"repeated sentences ({rep.repetition_ratio:.0%})")
    rep.ngram_loop_ratio = ngram_loop_ratio(text_all)
    if rep.ngram_loop_ratio > max_ngram_loop:
        rep.add(f"n-gram loop ({rep.ngram_loop_ratio:.0%} repeated 6-grams)")
    if _ROLE_ECHO.search(parsed.answer):
        rep.role_echo = True
        rep.add("role echo in answer text")
    norm_claims = [_norm(c.text) for c in parsed.claims]
    rep.duplicate_claims = len(norm_claims) - len(set(norm_claims))
    if rep.duplicate_claims:
        rep.add(f"{rep.duplicate_claims} duplicate claim(s)")
    # every answer sentence should be backed by a claim (otherwise it escapes verification)
    claim_blob = " ".join(norm_claims)
    uncovered = 0
    for s, e in split_sentences(parsed.answer):
        sent = _norm(parsed.answer[s:e])
        if len(sent) < 20:
            continue
        words = sent.split()
        hits = sum(1 for w in words if w in claim_blob)
        if hits / max(1, len(words)) < 0.5:
            uncovered += 1
    rep.uncovered_sentences = uncovered
    if uncovered > max_uncovered:
        rep.add(f"{uncovered} answer sentence(s) not covered by any claim")
    if question_entities and claim_entities is not None and parsed.claims:
        if not (question_entities & claim_entities):
            rep.off_topic = True
            rep.add("no claim mentions any entity from the question")
    return rep


def feedback_for(rep: QualityReport, verification_notes: list[str] | None = None) -> str:
    """Precise, evidence-bound feedback for the next attempt (never new facts)."""
    lines = ["Your previous output was rejected for these reasons:"]
    lines += [f"- {d}" for d in rep.defects]
    lines += [f"- {n}" for n in (verification_notes or [])]
    lines.append("Produce the JSON object again from the same evidence only: no repeated "
                 "sentences, each claim once, every answer sentence backed by a claim with its "
                 "citation ids, and abstain if the evidence does not support an answer.")
    return "\n".join(lines)
