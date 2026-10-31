"""Entity consistency: every entity named in a claim must appear in the evidence that supports it.

This replaces the thesis's ``Signal_Entity`` (regex terms from the *query* checked against the
*context*), which could not catch an entity substitution in the answer. The extractor is a
protocol: ``RegexBiomedEntityExtractor`` is the dependency-free default; a scispaCy/UMLS adapter
(kind ``scispacy``) can be dropped in through the registry.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.core import symbols


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
    return symbols.normalize(e)


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


# ------------------------------------------------------------------ query grounding -------------
@dataclass(frozen=True)
class QueryGrounding:
    question_entities: tuple[str, ...]
    missing_in_evidence: tuple[str, ...]  # question entities absent from all supporting texts
    missing_in_critical_claims: tuple[str, ...]  # question entities no critical claim mentions
    substitutions: dict[str, tuple[str, str]]  # claim_id -> (claim entity, question entity)

    @property
    def grounding(self) -> float | None:
        n = len(self.question_entities)
        return None if n == 0 else (n - len(self.missing_in_evidence)) / n

    @property
    def substitution_detected(self) -> bool:
        return bool(self.substitutions)


def query_entity_grounding(
    question: str,
    claims: Sequence[tuple[str, str, bool]],  # (claim_id, text, critical)
    supporting_texts: Iterable[str],
    extractor: EntityExtractor,
) -> QueryGrounding:
    """Ground the *question's* entities, not only the claims'.

    Catches the case the per-claim check cannot: a claim about MAP3K7, correctly cited to a
    MAP3K7 passage, answering a MAP3K3 question. Every entity of a critical claim that is a near
    neighbour of a question entity (same stem, small edit distance, or same drug class suffix)
    while the question entity itself is absent from that claim is a substitution.
    """
    q_ents = {symbols.normalize(e) for e in extractor.extract(question)}
    q_ents |= {symbols.normalize(t) for t in symbols.symbol_like_tokens(question)}
    q_ents.discard("")
    if not q_ents:
        return QueryGrounding((), (), (), {})
    texts = list(supporting_texts)
    pool: set[str] = set()
    for t in texts:
        pool |= {symbols.normalize(e) for e in extractor.extract(t)}
    missing_ev = tuple(sorted(e for e in q_ents if not _present(e, pool, texts)))

    critical_ents: set[str] = set()
    subs: dict[str, tuple[str, str]] = {}
    for cid, text, critical in claims:
        if not critical:
            continue
        c_ents = {symbols.normalize(e) for e in extractor.extract(text)}
        c_ents |= {symbols.normalize(t) for t in symbols.symbol_like_tokens(text)}
        critical_ents |= c_ents
        for ce in c_ents:
            for qe in q_ents:
                if qe not in c_ents and symbols.is_near_neighbour(ce, qe):
                    subs[cid] = (ce, qe)
    missing_cc = tuple(sorted(e for e in q_ents if e not in critical_ents)) if claims else ()
    return QueryGrounding(tuple(sorted(q_ents)), missing_ev, missing_cc, subs)
