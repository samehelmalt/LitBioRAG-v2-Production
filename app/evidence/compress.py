"""Evidence compression: keep the part of each passage the question is about.

The generator sees a contiguous window of each passage — from the first to the last sentence
that shares content words (or an exact symbol) with the question, plus ``context`` neighbouring
sentences — with ``char_start``/``char_end`` adjusted so citations still point into the source.
A passage with no overlapping sentence is kept whole up to ``max_sentences`` (it was retrieved
for a reason the lexical overlap cannot see). The verifier always sees the full passage.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.core.symbols import normalize, symbol_like_tokens
from app.evidence.selector import EvidenceUnit
from app.retrieval.passages import split_sentences

_WORD = re.compile(r"[a-z0-9][a-z0-9-]{2,}")
_STOP = {"the", "and", "for", "with", "that", "this", "what", "which", "does", "are", "was",
         "were", "from", "into", "than", "have", "has", "had", "not", "its", "their", "about",
         "role", "effect", "effects", "latest", "approaches", "treatment", "between", "among",
         "how", "why", "when", "where", "can", "could", "should", "would", "patients", "study"}


def _content_words(text: str) -> set[str]:
    words = {w for w in _WORD.findall(text.lower()) if w not in _STOP}
    words |= {normalize(s) for s in symbol_like_tokens(text)}
    return words


def compress_unit(u: EvidenceUnit, question: str, *, max_sentences: int = 4,
                  context: int = 1, min_overlap: int = 1) -> EvidenceUnit:  # fmt: skip
    spans = split_sentences(u.text)
    if len(spans) <= max_sentences:
        return u
    q = _content_words(question)
    hits = [i for i, (s, e) in enumerate(spans)
            if len(_content_words(u.text[s:e]) & q) >= min_overlap]  # fmt: skip
    if not hits:
        lo, hi = 0, max_sentences - 1
    else:
        lo, hi = max(0, hits[0] - context), min(len(spans) - 1, hits[-1] + context)
        if hi - lo + 1 > max_sentences:  # centre on the densest run of hits
            best, best_n = lo, -1
            for start in range(lo, hi - max_sentences + 2):
                n = sum(1 for h in hits if start <= h < start + max_sentences)
                if n > best_n:
                    best, best_n = start, n
            lo, hi = best, best + max_sentences - 1
    s, e = spans[lo][0], spans[hi][1]
    text = u.text[s:e].strip()
    lead = u.text[s:e].find(text)
    return u.model_copy(update={"text": text, "char_start": u.char_start + s + lead,
                                "char_end": u.char_start + s + lead + len(text)})  # fmt: skip


def compress_evidence(units: Sequence[EvidenceUnit], question: str, **kw) -> list[EvidenceUnit]:
    return [compress_unit(u, question, **kw) for u in units]
