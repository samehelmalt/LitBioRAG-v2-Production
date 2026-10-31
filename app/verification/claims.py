"""Claim preparation for verification.

Claims come structured from the generator contract. This module normalizes them for checking:
splits any multi-sentence claim text into atomic sentences (keeping the citations), fills the
``critical`` flag when the generator left it unset, and extracts the negation polarity used by the
contradiction check. The thesis's regex bullet splitter is not reproduced; structure comes from
the contract, the sentence splitter is only a safety net.
"""

from __future__ import annotations

import re

from app.generation.contract import Claim, GeneratedAnswer
from app.retrieval.passages import split_sentences

_NUMERIC = re.compile(r"\d")
_POLARITY = re.compile(
    r"\b(yes|no|not|never|increase[sd]?|decrease[sd]?|reduce[sd]?|improve[sd]?|prevent[sd]?|"
    r"cause[sd]?|associated|required|effective|superior|inferior|significant(?:ly)?)\b",
    re.IGNORECASE,
)
_HEDGE = re.compile(r"\b(may|might|could|possibly|suggest(?:s|ed)?|appear(?:s|ed)?)\b", re.I)


def is_critical(text: str) -> bool:
    """A claim carries the answer when it states a number, a polarity or an effect direction."""
    return bool(_NUMERIC.search(text) or _POLARITY.search(text))


def modality(text: str) -> str:
    return "hedged" if _HEDGE.search(text) else "asserted"


def prepare_claims(answer: GeneratedAnswer, *, max_claims: int = 20) -> list[Claim]:
    out: list[Claim] = []
    for c in answer.claims:
        spans = split_sentences(c.text)
        parts = [c.text[s:e].strip() for s, e in spans] if len(spans) > 1 else [c.text.strip()]
        for i, part in enumerate(parts):
            if len(part) < 8:
                continue
            cid = c.id if len(parts) == 1 else f"{c.id}.{i + 1}"
            out.append(
                c.model_copy(
                    update={
                        "id": cid,
                        "text": part,
                        "critical": c.critical or is_critical(part),
                        "modality": c.modality or modality(part),
                    }
                )
            )
            if len(out) >= max_claims:
                return out
    return out
