"""Entity consistency: every entity named in a claim must appear in the evidence that supports it.

This replaces the thesis's ``Signal_Entity`` (regex terms from the *query* checked against the
*context*), which could not catch an entity substitution in the answer. The extractor is a
protocol: ``RegexBiomedEntityExtractor`` is the dependency-free default; a scispaCy/UMLS adapter
(kind ``scispacy``) can be dropped in through the registry.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol


class EntityExtractor(Protocol):
    name: str

    def extract(self, text: str) -> set[str]: ...


_GENE = re.compile(
    r"\b(?=[A-Za-z0-9-]{3,12}\b)(?:[A-Z]+[0-9]+[A-Za-z0-9-]*|[A-Z]{2,}[a-z]?[0-9]*)\b"
)
_DRUG = re.compile(
    r"\b[A-Za-z]{4,}(?:mab|nib|pril|sartan|statin|olol|azole|cillin|mycin|cycline|vir|tide|"
    r"parin|prazole|dipine|triptan|afil|oxetine|ridine|ide|ine|one|ate|ium)\b",
    re.IGNORECASE,
)
_GREEK = re.compile(r"\b(?:NF-?kappa-?B|TNF-?alpha|IL-?\d+|TGF-?beta\d?|IFN-?gamma)\b", re.I)
_COMMON_UPPER = {"THE", "AND", "FOR", "WITH", "THIS", "THAT", "FROM", "WERE", "WAS", "NOT", "DNA",
                 "RNA", "USA", "CI", "HR", "OR", "RR", "SD", "SE", "N", "P", "VS", "ET", "AL",
                 "YES", "NO", "MRNA"}  # fmt: skip


def normalize_entity(e: str) -> str:
    return re.sub(r"[^a-z0-9]", "", e.lower())


class RegexBiomedEntityExtractor:
    """Gene symbols, acronyms, named pathways and drug-like tokens. Precision over recall."""

    name = "regex:biomed-v1"

    def extract(self, text: str) -> set[str]:
        found: set[str] = set()
        for m in _GREEK.finditer(text):
            found.add(normalize_entity(m.group(0)))
        text = _GREEK.sub(" ", text)  # do not re-match 'NF' / 'IL' as separate symbols
        for m in _GENE.finditer(text):
            tok = m.group(0)
            if tok.upper() in _COMMON_UPPER or tok.isdigit():
                continue
            found.add(normalize_entity(tok))
        for m in _DRUG.finditer(text):
            tok = m.group(0)
            if tok.lower() in _DRUG_STOP:
                continue
            found.add(normalize_entity(tok))
        return found


# frequent English words that end like drug names
_DRUG_STOP = {
    "outcome", "medicine", "determine", "examine", "combine", "baseline", "guideline", "routine",
    "decline", "define", "outline", "underline", "alone", "someone", "anyone", "everyone",
    "telephone", "bone", "zone", "done", "gone", "stone", "none", "estimate", "indicate",
    "evaluate", "associate", "demonstrate", "generate", "separate", "moderate", "accurate",
    "candidate", "appropriate", "adequate", "immediate", "intermediate", "provide", "inside",
    "outside", "decide", "guide", "wide", "aside", "beside", "oxide", "medium", "premium",
    "sodium", "calcium", "potassium", "magnesium", "lithium", "criterium", "criteria", "ovaries",
    "injuries", "studies", "series", "species", "theories", "varies", "hemophilia",
}


def entity_consistency(
    claim_text: str,
    supporting_texts: Iterable[str],
    extractor: EntityExtractor,
) -> tuple[int, int, tuple[str, ...]]:
    """Return (checked, ok, missing): entities in the claim absent from all supporting texts."""
    claim_ents = extractor.extract(claim_text)
    if not claim_ents:
        return 0, 0, ()
    pool: set[str] = set()
    for t in supporting_texts:
        pool |= extractor.extract(t)
        pool.add(normalize_entity(t))  # cheap containment fallback below
    missing = tuple(sorted(e for e in claim_ents if not _present(e, pool, supporting_texts)))
    return len(claim_ents), len(claim_ents) - len(missing), missing


def _present(ent: str, pool: set[str], texts: Iterable[str]) -> bool:
    if ent in pool:
        return True
    return any(ent in normalize_entity(t) for t in texts)
