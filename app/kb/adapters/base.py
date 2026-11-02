from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.documents import Document, FileMeta, SourceType


@dataclass(frozen=True)
class UnitSpec:
    """One citable unit extracted from a file, before it becomes a Document."""

    locator: str  # "p3" | "s7" | "Sheet1!A1:D40" | "sec2" | "t1"
    locator_kind: str  # page | slide | sheet | section | text
    label: str  # "page 3"
    title: str  # unit heading (page/section/sheet title)
    text: str
    page_number: int | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    heading: str = ""
    slide_number: int | None = None
    n_images: int = 0
    images_with_text: int = 0


class DocumentAdapter(Protocol):
    kind: str

    def extract(self, path: Path) -> Iterable[UnitSpec]: ...


@dataclass(frozen=True)
class LectureMeta:
    """Owner-supplied provenance for an uploaded deck (form fields on upload)."""

    course_id: str = ""
    lecture_id: str = ""
    lecture_title: str = ""
    academic_year: str = ""
    deck_version: str = ""

    @staticmethod
    def from_mapping(m: dict | None, *, file_name: str = "") -> LectureMeta:
        m = m or {}
        stem = Path(file_name).stem if file_name else ""
        lecture_id = str(m.get("lecture_id") or "").strip()
        return LectureMeta(
            course_id=str(m.get("course_id") or "").strip(),
            lecture_id=lecture_id or (re.sub(r"[^A-Za-z0-9_.]+", "-", stem).strip("-")
                                      if m.get("course_id") else ""),
            lecture_title=str(m.get("lecture_title") or stem).strip(),
            academic_year=str(m.get("academic_year") or "").strip(),
            deck_version=str(m.get("version") or m.get("deck_version") or "").strip(),
        )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_WS = re.compile(r"[ \t ]+")
_NL = re.compile(r"\n{3,}")


def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(_WS.sub(" ", line).strip() for line in s.split("\n"))
    return _NL.sub("\n\n", s).strip()


def units_to_documents(
    units: Iterable[UnitSpec], *, kb_id: str, file_id: str, file_name: str, kind: str,
    version: str = "", uploaded_by: str = "", min_chars: int = 20,
    lecture: LectureMeta | None = None,
) -> list[Document]:  # fmt: skip
    docs: list[Document] = []
    lec = lecture or LectureMeta()
    for u in units:
        text = clean_text(u.text)
        image_only = u.n_images > 0 and u.images_with_text == 0
        if len(text) < min_chars and not image_only:
            continue
        if len(text) < min_chars:  # keep the slide so the gap is visible and citable
            text = f"{u.title}\n(image-only slide: no text could be extracted from its "
            text += f"{u.n_images} image(s))"
        meta = FileMeta(
            kb_id=kb_id, file_id=file_id, file_name=file_name, kind=kind,
            locator_kind=u.locator_kind, locator=u.locator, label=u.label,
            page_number=u.page_number, sheet_name=u.sheet_name, cell_range=u.cell_range,
            heading=u.heading, version=version or lec.deck_version,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            uploaded_by=uploaded_by, course_id=lec.course_id, lecture_id=lec.lecture_id,
            lecture_title=lec.lecture_title, academic_year=lec.academic_year,
            deck_version=lec.deck_version, slide_number=u.slide_number, section=u.heading,
            n_images=u.n_images, images_with_text=u.images_with_text, needs_ocr=image_only,
        )  # fmt: skip
        docs.append(Document(
            doc_id=Document.make_file_id(file_id, u.locator),
            title=(u.title or f"{file_name} — {u.label}")[:300], abstract=text,
            source_type=(SourceType.COURSE_SLIDE if lec.lecture_id and kind == "pptx"
                         else SourceType.UPLOADED_DOCUMENT),
            source=f"upload:{kind}", journal=lec.lecture_title or None,
            year=int(lec.academic_year[:4]) if lec.academic_year[:4].isdigit() else None,
            version=(version or lec.deck_version) or None, file=meta,
        ))  # fmt: skip
    return docs
