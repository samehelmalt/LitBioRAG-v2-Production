"""Claim-level answer repair: drop unsupported non-critical claims, keep everything else verbatim.

No regeneration: the surviving claims are re-joined in their original order with their citations.
The removed claims are reported to the user, not hidden.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.generation.contract import GeneratedAnswer


def repair_answer(answer: GeneratedAnswer, remove_ids: Iterable[str]) -> tuple[str, list[str]]:
    remove = set(remove_ids)
    kept, removed = [], []
    for c in answer.claims:
        base_id = c.id.split(".", 1)[0]
        if c.id in remove or base_id in remove:
            removed.append(c.text)
            continue
        cits = f" [{', '.join(c.citations)}]" if c.citations else ""
        kept.append(c.text.rstrip() + cits)
    return " ".join(kept).strip(), removed
