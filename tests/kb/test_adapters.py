"""Every supported file type yields citable units with locators; documents carry provenance."""

import pytest

pytest.importorskip("pypdf")
pytest.importorskip("docx")
pytest.importorskip("openpyxl")
pytest.importorskip("pptx")

from app.core.documents import SourceType  # noqa: E402
from app.kb.adapters import SUPPORTED_EXTENSIONS, adapter_for, units_to_documents  # noqa: E402
from tests.kb.fixtures import make_docx, make_pdf, make_pptx, make_xlsx  # noqa: E402


def test_pdf_pages(tmp_path):
    p = make_pdf(tmp_path / "guide.pdf", ["Metformin lowers HbA1c by about 1 percent.",
                                           "Dapagliflozin reduces heart failure admissions."])
    units = list(adapter_for(p).extract(p))
    assert [u.locator for u in units] == ["p1", "p2"]
    assert units[0].page_number == 1 and "Metformin" in units[0].text
    assert units[1].label == "page 2"


def test_docx_sections_and_tables(tmp_path):
    p = make_docx(tmp_path / "notes.docx",
                  [("", ["Preamble text that is long enough to keep."]),
                   ("Dosing", ["Start 500 mg daily.", "Increase weekly."]),
                   ("Monitoring", ["Check renal function."])],
                  table=[["Drug", "Dose"], ["Metformin", "500 mg"]])  # fmt: skip
    units = list(adapter_for(p).extract(p))
    heads = [u.heading for u in units]
    assert heads == ["", "Dosing", "Monitoring"]
    assert units[1].locator == "sec2" and units[1].label == "section 'Dosing'"
    assert "Metformin | 500 mg" in units[2].text  # table rows land in the last section


def test_xlsx_rows_with_cell_ranges(tmp_path):
    rows = [["Drug", "LDL reduction %"]] + [[f"statin{i}", 30 + i] for i in range(1, 61)]
    p = make_xlsx(tmp_path / "trials.xlsx", {"Lipids": rows, "Empty": []})
    units = list(adapter_for(p).extract(p))
    assert len(units) == 2  # 60 rows → 40 + 20
    assert units[0].sheet_name == "Lipids" and units[0].cell_range == "A2:B41"
    assert units[0].locator == "Lipids!A2:B41" and units[0].label == "Lipids rows 2–41"
    assert "Drug: statin1; LDL reduction %: 31" in units[0].text
    assert units[1].cell_range == "A42:B61"


def test_pptx_slides(tmp_path):
    p = make_pptx(tmp_path / "lecture.pptx", [("Hyperkalemia", ["Peaked T waves", "Calcium first"]),
                                              ("Hypokalemia", ["U waves"])])  # fmt: skip
    units = list(adapter_for(p).extract(p))
    assert [u.locator for u in units] == ["s1", "s2"] and "Peaked T waves" in units[0].text


def test_markdown_sections(tmp_path):
    p = tmp_path / "readme.md"
    p.write_text("intro line long enough\n# Dosing\nStart low.\n## Titration\nGo slow.\n")
    units = list(adapter_for(p).extract(p))
    assert [u.heading for u in units] == ["", "Dosing", "Titration"]
    assert units[1].locator == "t2"


def test_units_become_documents_with_provenance(tmp_path):
    p = make_pdf(tmp_path / "g.pdf", ["Metformin lowers HbA1c by about 1 percent.", "x"])
    docs = units_to_documents(adapter_for(p).extract(p), kb_id="kb_1", file_id="abcd1234",
                              file_name="g.pdf", kind="pdf", uploaded_by="u1")  # fmt: skip
    assert len(docs) == 1  # the 1-char page is dropped
    d = docs[0]
    assert d.doc_id == "file:abcd1234:p1" and d.source_type == SourceType.UPLOADED_DOCUMENT
    assert d.course_id == "kb_1" and d.file.label == "page 1" and not d.is_peer_reviewed


def test_unsupported_extension():
    with pytest.raises(ValueError):
        adapter_for("x.exe")
    assert ".pdf" in SUPPORTED_EXTENSIONS and ".xlsx" in SUPPORTED_EXTENSIONS
