"""PPTX → one unit per slide (title, body, tables, notes), via the course-deck extractor."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from app.kb.adapters.base import UnitSpec


class PptxAdapter:
    kind = "pptx"

    def extract(self, path: Path) -> Iterable[UnitSpec]:
        from scripts.corpus.pptx_decks import extract_deck

        docs = extract_deck(path, course_id="-", lecture_id="-", lecture_title=path.stem)
        for d in docs:
            s = d.slide
            n = s.slide_number if s else 0
            title = s.slide_title if s and s.slide_title else f"{path.stem} — slide {n}"
            yield UnitSpec(locator=f"s{n}", locator_kind="slide", label=f"slide {n}", title=title,
                           text=d.abstract, heading=s.section if s else "")  # fmt: skip
