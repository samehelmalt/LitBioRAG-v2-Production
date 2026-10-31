"""File adapters: one ``Document`` per citable unit (page, section, sheet block, slide)."""

from __future__ import annotations

from pathlib import Path

from app.kb.adapters.base import DocumentAdapter, UnitSpec, file_sha256, units_to_documents

_BY_EXT: dict[str, str] = {
    ".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx", ".xlsm": "xlsx", ".pptx": "pptx",
    ".txt": "text", ".md": "text", ".markdown": "text",
}  # fmt: skip
SUPPORTED_EXTENSIONS = tuple(_BY_EXT)


def adapter_for(path: Path | str) -> DocumentAdapter:
    kind = _BY_EXT.get(Path(path).suffix.lower())
    if kind is None:
        raise ValueError(f"unsupported file type {Path(path).suffix!r}; "
                         f"supported: {', '.join(SUPPORTED_EXTENSIONS)}")  # fmt: skip
    if kind == "pdf":
        from app.kb.adapters.pdf import PdfAdapter

        return PdfAdapter()
    if kind == "docx":
        from app.kb.adapters.docx import DocxAdapter

        return DocxAdapter()
    if kind == "xlsx":
        from app.kb.adapters.xlsx import XlsxAdapter

        return XlsxAdapter()
    if kind == "pptx":
        from app.kb.adapters.pptx import PptxAdapter

        return PptxAdapter()
    from app.kb.adapters.text import TextAdapter

    return TextAdapter()


__all__ = ["DocumentAdapter", "UnitSpec", "adapter_for", "file_sha256", "units_to_documents",
           "SUPPORTED_EXTENSIONS"]  # fmt: skip
