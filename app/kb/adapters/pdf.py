"""PDF → one unit per page (pypdf). Scanned pages yield no text; an OCR hook can fill them."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from app.kb.adapters.base import UnitSpec

OcrHook = Callable[[Path, int], str]  # (file, page_number) -> text


class PdfAdapter:
    kind = "pdf"

    def __init__(self, ocr: OcrHook | None = None):
        self.ocr = ocr

    def extract(self, path: Path) -> Iterable[UnitSpec]:
        from pypdf import PdfReader  # optional dependency (kb extra)

        reader = PdfReader(str(path))
        title = ""
        try:
            title = (reader.metadata.title or "").strip() if reader.metadata else ""
        except Exception:  # noqa: BLE001  (malformed metadata is not an extraction failure)
            title = ""
        for n, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                text = ""
            if not text.strip() and self.ocr is not None:
                text = self.ocr(path, n)
            yield UnitSpec(locator=f"p{n}", locator_kind="page", label=f"page {n}",
                           title=f"{title or path.name} — page {n}", text=text,
                           page_number=n)  # fmt: skip
