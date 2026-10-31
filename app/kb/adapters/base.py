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


class DocumentAdapter(Protocol):
    kind: str

    def extract(self, path: Path) -> Iterable[UnitSpec]: ...


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
) -> list[Document]:  # fmt: skip
    docs: list[Document] = []
    for u in units:
        text = clean_text(u.text)
        if len(text) < min_chars:
            continue
        meta = FileMeta(
            kb_id=kb_id, file_id=file_id, file_name=file_name, kind=kind,
            locator_kind=u.locator_kind, locator=u.locator, label=u.label,
            page_number=u.page_number, sheet_name=u.sheet_name, cell_range=u.cell_range,
            heading=u.heading, version=version,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            uploaded_by=uploaded_by,
        )  # fmt: skip
        docs.append(Document(
            doc_id=Document.make_file_id(file_id, u.locator),
            title=(u.title or f"{file_name} — {u.label}")[:300], abstract=text,
            source_type=SourceType.UPLOADED_DOCUMENT, source=f"upload:{kind}",
            version=version or None, file=meta,
        ))  # fmt: skip
    return docs
