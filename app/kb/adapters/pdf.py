"""PDF → one unit per page (pypdf). Scanned pages yield no text; an OCR hook can fill them."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from app.kb.adapters.base import UnitSpec

OcrHook = Callable[[Path, int], str]  # (file, page_number) -> text


class PdfAdapter:
    kind = "pdf"

    def __init__(self, ocr: OcrHook | None = None, image_text=None):
        self.ocr = ocr
        self.image_text = image_text  # app.kb.vision.ImageTextExtractor

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
            n_images = images_with_text = 0
            if not text.strip():
                try:
                    images = list(page.images)
                except Exception:  # noqa: BLE001
                    images = []
                n_images = len(images)
                if self.image_text is not None:
                    for i, img in enumerate(images, start=1):
                        try:
                            t = (self.image_text.extract(img.data, context=f"page {n}") or "")
                        except Exception:  # noqa: BLE001
                            t = ""
                        if t.strip():
                            images_with_text += 1
                            text += f"\n[image {i} text] {t.strip()}"
            yield UnitSpec(locator=f"p{n}", locator_kind="page", label=f"page {n}",
                           title=f"{title or path.name} — page {n}", text=text,
                           page_number=n, n_images=n_images,
                           images_with_text=images_with_text)  # fmt: skip
