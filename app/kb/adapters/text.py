"""TXT / Markdown → one unit per heading section (or fixed-size blocks when no headings)."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from app.kb.adapters.base import UnitSpec

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")


class TextAdapter:
    kind = "text"

    def __init__(self, block_chars: int = 4000):
        self.block_chars = block_chars

    def extract(self, path: Path) -> Iterable[UnitSpec]:
        text = path.read_text(encoding="utf-8", errors="replace")
        sections: list[tuple[str, list[str]]] = [("", [])]
        for line in text.splitlines():
            m = _HEADING.match(line)
            if m:
                sections.append((m.group(2).strip(), []))
            else:
                sections[-1][1].append(line)
        n = 0
        for heading, lines in sections:
            body = "\n".join(lines).strip()
            if not body:
                continue
            for start in range(0, len(body), self.block_chars):
                n += 1
                label = f"section '{heading}'" if heading else f"part {n}"
                yield UnitSpec(locator=f"t{n}", locator_kind="section" if heading else "text",
                               label=label, title=heading or f"{path.name} — part {n}",
                               text=body[start : start + self.block_chars],
                               heading=heading)  # fmt: skip
