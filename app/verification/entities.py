"""Entity consistency and question-entity grounding.

Replaces the thesis's ``Signal_Entity`` (regex terms from the *query* checked against the
*context*), which could not catch an entity substitution in the answer. The extractor is a
protocol: ``RegexBiomedEntityExtractor`` is the dependency-free default and covers gene/protein
symbols, drug names, conditions (suffix- and head-based, numbered families such as "type 2
diabetes" or "hepatitis B") and named pathways; a scispaCy/UMLS adapter (kind ``scispacy``) can be
dropped in through the registry. Which entities count as *confusable* is decided by
``app.core.symbols`` from ``configs/entities.yaml``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from app.core import symbols


class EntityExtractor(Protocol):
    name: str

    def extract(self, text: str) -> set[str]: ...


@dataclass(frozen=True)
class Mention:
    surface: str
    norm: str
    start: int
    end: int
    kind: str  # gene | drug | condition | pathway | code


_GENE = re.compile(
    r"\b(?=[A-Za-z0-9-]{3,12}\b)(?:[A-Z]+[0-9]+[A-Za-z0-9-]*|[A-Z]{2,}[a-z]?[0-9]*)\b"
)
_GREEK = re.compile(r"\b(?:NF-?kappa-?B|TNF-?alpha|IL-?\d+|TGF-?beta\d?|IFN-?gamma)\b", re.I)
_COMMON_UPPER = {"THE", "AND", "FOR", "WITH", "THIS", "THAT", "FROM", "WERE", "WAS", "NOT", "DNA",
                 "RNA", "USA", "CI", "HR", "OR", "RR", "SD", "SE", "N", "P", "VS", "ET", "AL",
                 "YES", "NO", "MRNA", "BMI", "ICU", "ED", "COVID", "ECG", "EKG", "MRI", "CT",
                 "PET", "EEG", "EMG", "QT", "QRS", "BP", "WHO", "FDA", "NIH", "CDC", "RCT", "EU",
                 "UK", "US", "QALY", "AUC"}  # measurement / agency acronyms: not entities
_STOP = {"the", "a", "an", "of", "in", "and", "or", "with", "for", "to", "this", "that", "these",
         "those", "new", "suspected", "confirmed", "known", "any", "all", "no", "is", "are", "was",
         "were",
         "patients", "patient", "risk", "treatment", "role", "effect", "effects", "cause",
         "causes", "involved", "between", "among", "about", "does", "do", "what", "which", "how"}
_TYPED = ("type", "stage", "grade", "class")  # "<fam> N <disease>" is one condition entity
_CONDITION_FAMILIES = {"type", "stage", "grade", "class", "trisomy", "hepatitis", "influenza",
                       "hpv", "herpes"}  # fmt: skip
_NUMERAL = re.compile(r"^(?:[0-9]+[a-z]?|I{1,3}|IV|V|VI{1,3}|IX|X)$", re.I)
_GENERIC_DRUG_TAIL = "ridine|ide|ine|one|ate|ium"
_DRUG_STOP = {
    "outcome", "medicine", "determine", "examine", "combine", "baseline", "guideline", "routine",
    "decline", "define", "outline", "underline", "alone", "someone", "anyone", "everyone",
    "telephone", "bone", "zone", "done", "gone", "stone", "none", "estimate", "indicate",
    "evaluate", "associate", "demonstrate", "generate", "separate", "moderate", "accurate",
    "candidate", "appropriate", "adequate", "immediate", "intermediate", "provide", "inside",
    "outside", "decide", "guide", "wide", "aside", "beside", "oxide", "medium", "premium",
    "sodium", "calcium", "potassium", "magnesium", "lithium", "criterium", "criteria", "ovaries",
    "injuries", "studies", "series", "species", "theories", "varies", "hemophilia", "vaccine",
    "cytokine", "creatinine", "cholesterol", "elevate", "mediate", "ultimate", "climate",
    "initiate", "alleviate", "reiterate", "deliberate", "pyruvate", "substrate", "phosphate",
    "sulfate", "nitrate", "lactate", "acetate", "glutamate", "folate", "hormone", "ketone",
    "backbone", "milestone", "ozone", "cortisone", "machine", "discipline", "turbine", "marine",
    "shine", "spine", "urine", "feline", "canine", "bovine", "murine", "porcine", "glycine",
    "alanine", "leucine", "lysine", "arginine", "cysteine", "histamine", "dopamine",
}
_COND_STOP = {"academia", "diagnosis", "prognosis", "apoptosis", "mitosis", "meiosis",
              "phagocytosis", "endocytosis", "exocytosis", "symbiosis", "attention", "extension",
              "retention", "intention", "detention", "contention", "pretension", "diploma",
              "aroma", "stoma", "genoma", "osmosis", "hypothesis", "homeostasis", "sclerosis",
              "entropy", "atrophy", "philosophy", "geography", "biography", "anatomy", "autonomy",
              "economy", "taxonomy", "dichotomy"}  # fmt: skip
# words that are legitimately both a drug-stop and a real drug are resolved in favour of the drug
_DRUG_STOP -= {"dopamine", "histamine", "cortisone"}


def normalize_entity(e: str) -> str:
    return symbols.normalize(e)


class RegexBiomedEntityExtractor:
    """Genes/proteins, drugs, conditions, pathways, codes. Precision over recall; rules in YAML."""

    name = "regex:biomed-v2"

    def __init__(self) -> None:
        r = symbols.rules()
        heads = "|".join(map(re.escape, r.disease_heads)) or "(?!x)x"
        fams = "|".join(map(re.escape, r.numbered_families)) or "(?!x)x"
        csuf = "|".join(map(re.escape, r.disease_suffixes)) or "(?!x)x"
        dsuf = "|".join(map(re.escape, r.drug_suffixes)) or "(?!x)x"
        num = r"(?P<n>[0-9]+[a-z]?|[A-Z]\b|(?:I{1,3}|IV|V|VI{1,3}|IX|X)\b)"
        typed = "|".join(f for f in _TYPED if f in r.numbered_families) or "(?!x)x"
        self._numbered = re.compile(rf"\b(?P<fam>{fams})[-\s]?{num}", re.I)
        self._typed = re.compile(
            rf"\b(?P<fam>{typed})[-\s]?{num}\s(?P<head>{heads})(?:\smellitus)?", re.I
        )
        self._headed = re.compile(
            rf"\b(?P<pre>(?:[A-Za-z][A-Za-z'’-]{{2,}}\s){{1,3}})(?P<head>{heads})\b", re.I
        )
        self._cond_suffix = re.compile(rf"\b[a-z]{{3,}}(?:{csuf})\b", re.I)
        self._drug = re.compile(
            rf"\b[A-Za-z]{{4,}}(?:{dsuf}|{_GENERIC_DRUG_TAIL})\b", re.I
        )
        self._code = symbols._CODE

    # ----------------------------------------------------------------------------------------
    def mentions(self, text: str) -> list[Mention]:
        found: list[Mention] = []

        def add(surface: str, start: int, kind: str) -> None:
            n = normalize_entity(surface)
            if n:
                found.append(Mention(surface, n, start, start + len(surface), kind))

        for m in self._typed.finditer(text):
            add(m.group(0).strip(), m.start(), "condition")
        for m in self._numbered.finditer(text):
            kind = "condition" if m.group("fam").lower() in _CONDITION_FAMILIES else "gene"
            add(m.group(0).strip(), m.start(), kind)
        for m in self._headed.finditer(text):
            raw_words = m.group("pre").split()
            nums = [i for i, w in enumerate(raw_words) if _NUMERAL.match(w)]
            if nums:  # "stage III colon cancer" → "colon cancer" (the stage is its own entity)
                raw_words = raw_words[nums[-1] + 1 :]
            words = [w for w in raw_words if w.lower().strip("'’") not in _STOP]
            if not words:
                continue
            surface = " ".join(words + [m.group("head")])
            add(surface, m.end() - len(surface), "condition")
        for m in self._cond_suffix.finditer(text):
            if m.group(0).lower() not in _COND_STOP:
                add(m.group(0), m.start(), "condition")
        for m in _GREEK.finditer(text):
            add(m.group(0), m.start(), "pathway")
        for m in self._code.finditer(text):
            add(m.group(0), m.start(), "code")
        for m in _GENE.finditer(text):
            tok = m.group(0)
            if tok.upper() in _COMMON_UPPER or tok.isdigit():
                continue
            add(tok, m.start(), "gene")
        for m in self._drug.finditer(text):
            if m.group(0).lower() not in _DRUG_STOP:
                add(m.group(0), m.start(), "drug")
        return _drop_nested(found)

    def extract(self, text: str) -> set[str]:
        return {m.norm for m in self.mentions(text)}


def _drop_nested(ms: list[Mention]) -> list[Mention]:
    """Keep the longest mention for overlapping spans ('hepatitis B' beats 'hepatitis')."""
    ms = sorted(ms, key=lambda m: (m.start, -(m.end - m.start)))
    out: list[Mention] = []
    for m in ms:
        if any(o.start <= m.start and m.end <= o.end and o is not m for o in out):
            continue
        out.append(m)
    seen: set[tuple[str, int]] = set()
    uniq = []
    for m in out:
        if (m.norm, m.start) not in seen:
            seen.add((m.norm, m.start))
            uniq.append(m)
    return uniq


def entity_consistency(
    claim_text: str,
    supporting_texts: Iterable[str],
    extractor: EntityExtractor,
) -> tuple[int, int, tuple[str, ...]]:
    """Return (checked, ok, missing): entities in the claim absent from all supporting texts."""
    claim_ents = extractor.extract(claim_text)
    if not claim_ents:
        return 0, 0, ()
    texts = list(supporting_texts)
    pool: set[str] = set()
    for t in texts:
        pool |= extractor.extract(t)
    missing = tuple(sorted(e for e in claim_ents if not _present(e, pool, texts)))
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
    off_target: dict[str, tuple[str, str]]  # neighbour named; asked entity covered elsewhere
    reasons: dict[str, str] = field(default_factory=dict)  # claim_id -> neighbour reason

    @property
    def grounding(self) -> float | None:
        n = len(self.question_entities)
        return None if n == 0 else (n - len(self.missing_in_evidence)) / n

    @property
    def substitution_detected(self) -> bool:
        return bool(self.substitutions)


def _entities(text: str, extractor: EntityExtractor) -> set[str]:
    ents = {symbols.normalize(e) for e in extractor.extract(text)}
    ents |= {symbols.normalize(t) for t in symbols.symbol_like_tokens(text)}
    ents.discard("")
    return ents


def query_entity_grounding(
    question: str,
    claims: Sequence[tuple[str, str, bool]],  # (claim_id, text, critical)
    supporting_texts: Iterable[str],
    extractor: EntityExtractor,
) -> QueryGrounding:
    """Ground the *question's* entities, not only the claims'.

    Catches the case the per-claim check cannot: a claim about MAP3K7, correctly cited to a
    MAP3K7 passage, answering a MAP3K3 question — and likewise simvastatin for atorvastatin,
    hyperkalemia for hypokalemia, type 1 for type 2 diabetes, hepatitis C for hepatitis B. A
    claim naming a near neighbour (``symbols.neighbour_reason``) of a question entity without
    naming the asked entity is a **substitution** when no critical claim names the asked entity
    at all, and an **off-target** claim when the answer does cover the asked entity elsewhere
    (the gate removes it rather than abstaining).
    """
    q_ents = _entities(question, extractor)
    if not q_ents:
        return QueryGrounding((), (), (), {}, {})
    texts = list(supporting_texts)
    pool: set[str] = set()
    for t in texts:
        pool |= _entities(t, extractor)
    missing_ev = tuple(sorted(e for e in q_ents if not _present(e, pool, texts)))

    critical_ents: set[str] = set()
    neighbours: dict[str, tuple[str, str, bool, str]] = {}
    for cid, text, critical in claims:
        c_ents = _entities(text, extractor)
        if critical:
            critical_ents |= c_ents
        for ce in c_ents:
            for qe in q_ents:
                reason = symbols.neighbour_reason(ce, qe) if qe not in c_ents else None
                if reason:
                    neighbours[cid] = (ce, qe, critical, reason)
    missing_cc = tuple(sorted(e for e in q_ents if e not in critical_ents)) if claims else ()
    subs = {cid: (ce, qe) for cid, (ce, qe, crit, _) in neighbours.items()
            if crit and qe in missing_cc}  # fmt: skip
    off = {cid: (ce, qe) for cid, (ce, qe, _, _) in neighbours.items() if cid not in subs}
    reasons = {cid: r for cid, (_, _, _, r) in neighbours.items()}
    return QueryGrounding(tuple(sorted(q_ents)), missing_ev, missing_cc, subs, off, reasons)
