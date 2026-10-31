"""XLSX → units of up to ``rows_per_unit`` rows per sheet, rendered 'Header: value' so that a
claim can cite 'Sheet1 rows 2–41' and the cell range is kept for the viewer."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from app.kb.adapters.base import UnitSpec


def _col(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


class XlsxAdapter:
    kind = "xlsx"

    def __init__(self, rows_per_unit: int = 40, max_cols: int = 30):
        self.rows_per_unit = rows_per_unit
        self.max_cols = max_cols

    def extract(self, path: Path) -> Iterable[UnitSpec]:
        import openpyxl  # kb extra

        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        for ws in wb.worksheets:
            rows = [list(r)[: self.max_cols] for r in ws.iter_rows(values_only=True)]
            rows = [r for r in rows if any(c is not None and str(c).strip() for c in r)]
            if not rows:
                continue
            header = [str(c).strip() if c is not None else _col(i + 1)
                      for i, c in enumerate(rows[0])]  # fmt: skip
            body = rows[1:] if len(rows) > 1 else rows
            first_data_row = 2 if len(rows) > 1 else 1
            ncols = max(len(header), max((len(r) for r in body), default=0))
            for start in range(0, len(body), self.rows_per_unit):
                chunk = body[start : start + self.rows_per_unit]
                r0 = first_data_row + start
                r1 = r0 + len(chunk) - 1
                lines = []
                for r in chunk:
                    pairs = [
                        f"{header[i] if i < len(header) else _col(i + 1)}: {c}"
                        for i, c in enumerate(r) if c is not None and str(c).strip()
                    ]
                    if pairs:
                        lines.append("; ".join(pairs))
                cell_range = f"A{r0}:{_col(ncols)}{r1}"
                yield UnitSpec(
                    locator=f"{ws.title}!{cell_range}", locator_kind="sheet",
                    label=f"{ws.title} rows {r0}–{r1}", title=f"{path.name} — {ws.title}",
                    text="\n".join(lines), sheet_name=ws.title, cell_range=cell_range,
                )  # fmt: skip
