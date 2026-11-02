"""PPTX → one unit per slide (title, body, tables, notes, alt text, image text), via the
course-deck extractor. Images go through the configured ``ImageTextExtractor``; slides whose
pictures yield no text are reported with ``n_images > 0, images_with_text == 0``."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from app.kb.adapters.base import UnitSpec


class PptxAdapter:
    kind = "pptx"

    def __init__(self, image_text=None):
        self.image_text = image_text

    def extract(self, path: Path) -> Iterable[UnitSpec]:
        from scripts.corpus.pptx_decks import extract_deck

        docs = extract_deck(path, course_id="-", lecture_id="-", lecture_title=path.stem,
                            image_text=self.image_text)  # fmt: skip
        for d in docs:
            s = d.slide
            n = s.slide_number if s else 0
            title = s.slide_title if s and s.slide_title else f"{path.stem} — slide {n}"
            yield UnitSpec(locator=f"s{n}", locator_kind="slide", label=f"slide {n}", title=title,
                           text=d.abstract, heading=s.section if s else "", slide_number=n,
                           n_images=s.n_images if s else 0,
                           images_with_text=s.images_with_text if s else 0)  # fmt: skip
