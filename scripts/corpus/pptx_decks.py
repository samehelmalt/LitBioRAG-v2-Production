"""PowerPoint decks -> one Document per slide (course-material adapter).

    python -m scripts.corpus.pptx_decks extract --deck lectures/cardio_04.pptx \
        --course-id im-review-2027 --lecture-id cardio-04 --lecture-title "Atrial Fibrillation" \
        --academic-year 2026-2027 --version v1 --out data/decks/cardio-04.jsonl
    python -m scripts.corpus.pptx_decks diff data/decks/cardio-04.v1.jsonl \
        data/decks/cardio-04.v2.jsonl

Each slide becomes a ``Document`` with ``source_type = course_slide`` and ``SlideMeta`` provenance
(course, lecture, slide number, section, version, content hash). Extracted per slide: title, text
frames, tables (one row per line, cells joined by " | "), picture alt text, speaker notes. Images
themselves are not OCR'd here; an OCR hook can add text to ``extra_text`` later. ``diff`` compares
two extractions by content hash so an annual update re-indexes only changed slides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.core.documents import Document, SlideMeta, SourceType  # noqa: E402

_PICTURE = 13  # MSO_SHAPE_TYPE.PICTURE


def _shape_texts(shape) -> list[str]:
    out: list[str] = []
    if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        for para in shape.text_frame.paragraphs:
            t = "".join(r.text for r in para.runs).strip()
            if t:
                out.append(t)
    if getattr(shape, "has_table", False) and shape.has_table:
        for row in shape.table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                out.append(" | ".join(cells))
    if shape.shape_type == _PICTURE:
        try:
            descr = shape._element.nvPicPr.cNvPr.get("descr")  # alt text
        except AttributeError:
            descr = None
        if descr:
            out.append(f"Image: {descr.strip()}")
    if getattr(shape, "shape_type", None) == 6:  # group
        for sub in shape.shapes:
            out.extend(_shape_texts(sub))
    return out


def extract_deck(
    path: Path,
    *,
    course_id: str,
    lecture_id: str,
    lecture_title: str,
    academic_year: str = "",
    version: str = "",
) -> list[Document]:
    from pptx import Presentation  # optional dependency (pptx extra)

    prs = Presentation(str(path))
    docs: list[Document] = []
    section = ""
    for n, slide in enumerate(prs.slides, start=1):
        title_shape = slide.shapes.title
        title = (title_shape.text or "").strip() if title_shape is not None else ""
        parts: list[str] = []
        for shape in slide.shapes:
            if title_shape is not None and shape.shape_id == title_shape.shape_id:
                continue
            parts.extend(_shape_texts(shape))
        notes = ""
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = (slide.notes_slide.notes_text_frame.text or "").strip()
        body = "\n".join(parts)
        this_section = section
        if not body and title:  # a title-only slide is a section header for the slides after it
            section = title
        if notes:
            body = f"{body}\nSpeaker notes: {notes}".strip()
        content_hash = hashlib.sha1(f"{title}\n{body}".encode()).hexdigest()[:16]
        year = int(academic_year[:4]) if academic_year[:4].isdigit() else None
        docs.append(
            Document(
                doc_id=Document.make_slide_id(lecture_id, n),
                title=title or f"Slide {n}",
                abstract=body,
                source_type=SourceType.COURSE_SLIDE,
                year=year,
                journal=lecture_title,
                source="pptx",
                version=version or None,
                slide=SlideMeta(
                    course_id=course_id, lecture_id=lecture_id, lecture_title=lecture_title,
                    slide_number=n, file_name=path.name, slide_title=title, section=this_section,
                    academic_year=academic_year, version=version, content_hash=content_hash,
                ),  # fmt: skip
            )
        )
    return docs


def write_jsonl(docs: Iterable[Document], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(d.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[Document]:
    with open(path, encoding="utf-8") as fh:
        return [Document.model_validate_json(line) for line in fh if line.strip()]


def diff_decks(old: list[Document], new: list[Document]) -> dict:
    """Slide-level change set by content hash: what an annual update must re-index."""
    o = {d.slide.slide_number: d for d in old if d.slide}
    n = {d.slide.slide_number: d for d in new if d.slide}
    common = o.keys() & n.keys()
    changed = sorted(k for k in common if o[k].slide.content_hash != n[k].slide.content_hash)
    return {
        "added": sorted(n.keys() - o.keys()),
        "removed": sorted(o.keys() - n.keys()),
        "changed": changed,
        "unchanged": len(o.keys() & n.keys()) - len(changed),
        "reindex_doc_ids": [n[k].doc_id for k in sorted(set(changed) | (n.keys() - o.keys()))],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--deck", type=Path, required=True)
    p.add_argument("--course-id", required=True)
    p.add_argument("--lecture-id", required=True)
    p.add_argument("--lecture-title", required=True)
    p.add_argument("--academic-year", default="")
    p.add_argument("--version", default="")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("diff")
    p.add_argument("old", type=Path)
    p.add_argument("new", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        docs = extract_deck(a.deck, course_id=a.course_id, lecture_id=a.lecture_id,
                            lecture_title=a.lecture_title, academic_year=a.academic_year,
                            version=a.version)  # fmt: skip
        n = write_jsonl(docs, a.out)
        print(json.dumps({"slides": n, "out": str(a.out)}))
    else:
        print(json.dumps(diff_decks(read_jsonl(a.old), read_jsonl(a.new)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
