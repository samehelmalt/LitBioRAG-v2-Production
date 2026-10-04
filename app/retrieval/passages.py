"""Sentence-aware passage splitting with character offsets.

Abstracts are short (200-400 words), so most documents become 1-3 passages. The splitter packs
whole sentences up to ``target_words`` and repeats the last sentence of a passage at the start of
the next (``overlap_sentences``) so a claim that straddles a boundary is still recoverable. Title
is included in passage 0 so title-only matches are indexable. Offsets refer to ``Document.text``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from app.core.documents import Document, Passage

# Sentence boundary: . ! ? followed by whitespace and an uppercase letter, digit or bracket.
# Protects common biomedical abbreviations and decimals ("vs.", "e.g.", "Fig. 2", "p = 0.05").
_ABBREV = re.compile(
    r"\b(?:vs|e\.g|i\.e|etc|al|Fig|Figs|Dr|Prof|No|approx|ca|cf|resp|Ref|Refs|Eq|St|Mr|Mrs|Ms)\.$",
    re.IGNORECASE,
)
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of sentences in ``text``; never drops characters."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        candidate = text[start : m.start()]
        if _ABBREV.search(candidate.rstrip()) or re.search(r"\d\.$", candidate.rstrip()):
            continue  # abbreviation or decimal point: not a boundary
        spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _count_words(s: str) -> int:
    return len(s.split())


def split_document(
    doc: Document,
    *,
    target_words: int = 200,
    max_words: int = 300,
    overlap_sentences: int = 1,
) -> Iterator[Passage]:
    text = doc.text
    sents = split_sentences(text)
    if not sents:
        return
    ord_ = 0
    i = 0
    while i < len(sents):
        j = i
        words = 0
        while j < len(sents):
            w = _count_words(text[sents[j][0] : sents[j][1]])
            if j > i and (words + w > max_words or words >= target_words):
                break
            words += w
            j += 1
        start, end = sents[i][0], sents[j - 1][1]
        chunk = text[start:end]
        yield Passage(
            passage_id=Passage.make_id(doc.doc_id, ord_),
            doc_id=doc.doc_id,
            ord=ord_,
            text=chunk,
            char_start=start,
            char_end=end,
            n_words=_count_words(chunk),
        )
        ord_ += 1
        if j >= len(sents):
            break
        i = max(j - overlap_sentences, i + 1)
