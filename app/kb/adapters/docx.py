"""DOCX → one unit per heading section (python-docx); tables rendered as rows."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from app.kb.adapters.base import UnitSpec


class DocxAdapter:
    kind = "docx"

    def __init__(self, max_section_chars: int = 6000):
        self.max_section_chars = max_section_chars

    def extract(self, path: Path) -> Iterable[UnitSpec]:
        import docx  # python-docx (kb extra)
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        d = docx.Document(str(path))
        sections: list[tuple[str, list[str]]] = [("", [])]
        for block in d.element.body.iterchildren():
            tag = block.tag.rsplit("}", 1)[-1]
            if tag == "p":
                p = Paragraph(block, d)
                style = (p.style.name if p.style is not None else "") or ""
                text = p.text.strip()
                if style.lower().startswith("heading") and text:
                    sections.append((text, []))
                elif text:
                    sections[-1][1].append(text)
            elif tag == "tbl":
                t = Table(block, d)
                for row in t.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        sections[-1][1].append(" | ".join(cells))
        n = 0
        for heading, lines in sections:
            body = "\n".join(lines)
            if not body.strip():
                continue
            for start in range(0, len(body), self.max_section_chars):
                n += 1
                chunk = body[start : start + self.max_section_chars]
                label = f"section '{heading}'" if heading else f"section {n}"
                yield UnitSpec(locator=f"sec{n}", locator_kind="section", label=label,
                               title=heading or f"{path.name} — section {n}", text=chunk,
                               heading=heading)  # fmt: skip
