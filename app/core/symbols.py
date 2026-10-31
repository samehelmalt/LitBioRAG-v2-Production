"""Biomedical entity helpers shared by retrieval and verification (no model, no I/O at import).

An *entity* here is anything that must match exactly to count as the same thing: gene/protein
symbols (MAP3K3, TAK1, IL-6, CD4), drug names (atorvastatin), conditions (hypokalemia, type 2
diabetes, hepatitis B), trial/drug codes (TAS-102). *Near neighbours* (MAP3K3 vs MAP3K7,
atorvastatin vs simvastatin, hypo- vs hyperkalemia, type 1 vs type 2 diabetes) are different
things and must never be treated as matches or expansions. MAP3K3→MAP3K7 is one instance of a
general class; the rules live in ``configs/entities.yaml`` and are extended there, not in code.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _ROOT / "configs" / "entities.yaml"

_GENE_LIKE = re.compile(
    r"\b(?=[A-Za-z0-9-]{3,12}\b)(?:[A-Z]+[0-9]+[A-Za-z0-9-]*|[A-Z]{2,}[0-9]+)\b"
)
_CODE = re.compile(r"\b[A-Z]{2,5}-\d{2,5}\b")
_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7",
          "viii": "8", "ix": "9", "x": "10"}  # fmt: skip


class EntityRules:
    """Parsed ``configs/entities.yaml``. One instance per config path (cached)."""

    def __init__(self, raw: dict):
        self.synonyms: dict[str, str] = {_norm(k): _norm(v) for k, v in
                                         (raw.get("synonyms") or {}).items()}  # fmt: skip
        self.antonym_prefixes: list[tuple[str, str]] = [
            (a, b) for a, b in (raw.get("antonym_prefixes") or [])
        ]
        self.drug_suffixes: list[str] = sorted(
            (raw.get("drug_suffixes") or []), key=len, reverse=True
        )
        self.disease_suffixes: list[str] = list(raw.get("disease_suffixes") or [])
        self.disease_heads: list[str] = list(raw.get("disease_heads") or [])
        self.numbered_families: list[str] = list(raw.get("numbered_families") or [])
        self.groups: dict[str, int] = {}
        for i, group in enumerate(raw.get("confusable_groups") or []):
            for member in group:
                self.groups[self.canonical(_norm(member))] = i
        suf = "|".join(map(re.escape, self.drug_suffixes)) or "(?!x)x"
        self.drug_re = re.compile(rf"^(?P<stem>[a-z]{{2,}})(?P<suffix>{suf})$")

    def canonical(self, normalized: str) -> str:
        return self.synonyms.get(normalized, normalized)


@lru_cache(maxsize=4)
def _rules_for(path: str) -> EntityRules:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return EntityRules(raw)


def rules() -> EntityRules:
    return _rules_for(os.environ.get("LITBIORAG_ENTITIES_CONFIG") or str(_DEFAULT_CONFIG))


def _norm(e: str) -> str:
    return re.sub(r"[^a-z0-9]", "", e.lower())


def normalize(e: str) -> str:
    """Lower-case, strip non-alphanumerics, map roman numerals, apply synonyms.

    'Type II Diabetes' -> 'type2diabetes'; 'TAK1' -> 'map3k7'; 'heart attack' -> 'mi'
    """
    s = e.lower().strip()
    s = re.sub(
        r"\b([a-z]+)\s+(i{1,3}|iv|v|vi{1,3}|ix|x)\b",
        lambda m: f"{m.group(1)} {_ROMAN[m.group(2)]}", s,
    )  # fmt: skip
    return rules().canonical(_norm(s))


def symbol_like_tokens(text: str) -> list[str]:
    """Exact-match tokens in query order, de-duplicated, original surface form."""
    out: list[str] = []
    for m in list(_CODE.finditer(text)) + list(_GENE_LIKE.finditer(text)):
        tok = m.group(0)
        if tok.isdigit() or tok.upper() in {"DNA", "RNA", "MRNA", "CI", "HR", "OR", "RR"}:
            continue
        if tok not in out:
            out.append(tok)
    return out


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _alpha_stem(sym: str) -> str:
    return re.sub(r"[0-9]+$", "", sym)


def _digitless(s: str) -> str:
    return re.sub(r"[0-9]+", "", s)


def neighbour_reason(a: str, b: str) -> str | None:
    """Why two *different* entities are confusable, or None. Reasons are stable strings used in
    verdict notes and the UI: numbered_family | letter_variant | stem_edit | antonym_prefix |
    drug_class:<suffix> | confusable_group."""
    r = rules()
    na, nb = normalize(a), normalize(b)
    if na == nb or not na or not nb:
        return None
    if na.startswith(nb) or nb.startswith(na):
        return None  # generic vs specific form of the same thing (hepatitis / hepatitis B)
    if _digitless(na) == _digitless(nb) and (na.isalpha() or nb.isalpha()):
        return None  # COVID vs COVID-19: same name with and without a number
    if (_digitless(na) == _digitless(nb) and _digitless(na)
            and any(c.isdigit() for c in na) and any(c.isdigit() for c in nb)):
        return "numbered_family"
    if _levenshtein(na, nb) <= 1 and max(len(na), len(nb)) >= 4:
        return "letter_variant"
    if _alpha_stem(na) and _alpha_stem(na) == _alpha_stem(nb) and _levenshtein(na, nb) <= 2:
        return "stem_edit"
    for p, q in r.antonym_prefixes:
        for x, y in ((p, q), (q, p)):
            if na.startswith(x) and nb.startswith(y) and na[len(x):] == nb[len(y):]:
                return "antonym_prefix"
    ma, mb = r.drug_re.match(na), r.drug_re.match(nb)
    if ma and mb and ma.group("suffix") == mb.group("suffix"):
        return f"drug_class:{ma.group('suffix')}"
    ga, gb = r.groups.get(na), r.groups.get(nb)
    if ga is not None and ga == gb:
        return "confusable_group"
    return None


def is_near_neighbour(a: str, b: str) -> bool:
    return neighbour_reason(a, b) is not None
