"""Lecture QA question sets: the human-reviewed evaluation data for the education product.

One JSONL record per question::

    {"id": "q001", "question": "...", "type": "factual",
     "gold_slides": ["IM-2026/L03/s4"],          # canonical ids without version/hash suffix
     "gold_answer": "...", "follow_up_of": null, "answerable": true,
     "reviewer": "dr-x", "notes": ""}

Types: factual, explanatory, multi_slide, numeric, follow_up, unanswerable. ``gold_slides`` are
course/lecture/slide prefixes of ``FileMeta.canonical_id`` so the set survives re-uploads of
the same lecture (version and content hash are not part of the gold id). ``follow_up_of`` names
the question whose delivered answer provides the conversational context. ``answerable: false``
marks questions the deck cannot answer — the system must abstain or clarify on those.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

TYPES = ("factual", "explanatory", "multi_slide", "numeric", "follow_up", "unanswerable")


@dataclass
class LectureQuestion:
    id: str
    question: str
    type: str
    gold_slides: list[str] = field(default_factory=list)
    gold_answer: str = ""
    follow_up_of: str | None = None
    answerable: bool = True
    reviewer: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.type not in TYPES:
            raise ValueError(f"{self.id}: type must be one of {TYPES}")
        if self.type == "unanswerable":
            self.answerable = False
        if self.type == "follow_up" and not self.follow_up_of:
            raise ValueError(f"{self.id}: follow_up needs follow_up_of")
        if self.answerable and not self.gold_slides:
            raise ValueError(f"{self.id}: an answerable question needs gold_slides")

    def to_dict(self) -> dict:
        return asdict(self)


def slide_key(canonical_id: str) -> str:
    """'IM-2026/L03/s4/v2026a/abcd' → 'IM-2026/L03/s4' (what gold ids are compared against)."""
    parts = canonical_id.split("/")
    return "/".join(parts[:3]) if len(parts) >= 3 else canonical_id


def load_questions(path: Path | str) -> list[LectureQuestion]:
    out: list[LectureQuestion] = []
    seen: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = LectureQuestion(**json.loads(line))
        if q.id in seen:
            raise ValueError(f"duplicate question id {q.id}")
        seen.add(q.id)
        out.append(q)
    for q in out:
        if q.follow_up_of and q.follow_up_of not in seen:
            raise ValueError(f"{q.id}: follow_up_of {q.follow_up_of!r} is not in the set")
    return out


def write_questions(questions: Iterable[LectureQuestion], path: Path | str) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def summarize_set(questions: list[LectureQuestion]) -> dict:
    by_type = {t: sum(q.type == t for q in questions) for t in TYPES}
    return {"n": len(questions), "by_type": by_type,
            "reviewed": sum(1 for q in questions if q.reviewer),
            "unanswerable": sum(1 for q in questions if not q.answerable)}  # fmt: skip
