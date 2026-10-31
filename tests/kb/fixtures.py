"""Generated office-file fixtures (no binary files committed)."""

from __future__ import annotations

from pathlib import Path


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Minimal valid PDF with one Helvetica text line per page (pypdf extracts it)."""
    objs: list[bytes] = []

    def add(b: bytes) -> int:
        objs.append(b)
        return len(objs)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    kids_placeholder = add(b"")  # pages object, filled later
    for text in pages:
        esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 50 750 Td ({esc}) Tj ET".encode("latin-1", "replace")
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page = add(
            b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] /Contents %d 0 R "
            b"/Resources << /Font << /F1 %d 0 R >> >> >>" % (kids_placeholder, content, font)
        )
        page_ids.append(page)
    kids = b" ".join(b"%d 0 R" % p for p in page_ids)
    objs[kids_placeholder - 1] = (b"<< /Type /Pages /Kids [" + kids + b"] /Count %d >>"
                                  % len(page_ids))
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % kids_placeholder)
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, catalog, xref))
    path.write_bytes(bytes(out))
    return path


def make_docx(path: Path, sections: list[tuple[str, list[str]]],
              table: list[list[str]] | None = None) -> Path:  # fmt: skip
    import docx

    d = docx.Document()
    for heading, paras in sections:
        if heading:
            d.add_heading(heading, level=1)
        for p in paras:
            d.add_paragraph(p)
    if table:
        t = d.add_table(rows=0, cols=len(table[0]))
        for row in table:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = v
    d.save(str(path))
    return path


def make_xlsx(path: Path, sheets: dict[str, list[list]]) -> Path:
    import openpyxl

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(r)
    wb.save(str(path))
    return path


def make_pptx(path: Path, slides: list[tuple[str, list[str]]]) -> Path:
    from pptx import Presentation

    prs = Presentation()
    layout = prs.slide_layouts[1]
    for title, bullets in slides:
        s = prs.slides.add_slide(layout)
        s.shapes.title.text = title
        tf = s.placeholders[1].text_frame
        tf.text = bullets[0] if bullets else ""
        for b in bullets[1:]:
            tf.add_paragraph().text = b
    prs.save(str(path))
    return path
