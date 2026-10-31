"""Biomedical symbol helpers shared by retrieval and verification (no model, no I/O).

A *symbol* is a token that must match exactly: gene/protein symbols (MAP3K3, TAK1, p53, IL-6),
trial/drug codes (TAS-102), and drug names. Near neighbours (MAP3K3 vs MAP3K7, atorvastatin vs
simvastatin) are different things and must never be treated as matches or expansions.
"""

from __future__ import annotations

import re

_GENE_LIKE = re.compile(
    r"\b(?=[A-Za-z0-9-]{3,12}\b)(?:[A-Z]+[0-9]+[A-Za-z0-9-]*|[A-Z]{2,}[0-9]+)\b"
)
_DRUG_SUFFIX = re.compile(
    r"^(?P<stem>[a-z]{2,})(?P<suffix>mab|nib|pril|sartan|statin|olol|azole|cillin|mycin|cycline|"
    r"vir|tide|parin|prazole|dipine|triptan|afil|oxetine)$"
)
_CODE = re.compile(r"\b[A-Z]{2,5}-\d{2,5}\b")

# surface forms that denote the same entity; keep tiny and explicit (ontology lookup is Step 5e)
SYNONYMS: dict[str, str] = {
    "heartattack": "myocardialinfarction",
    "mi": "myocardialinfarction",
    "acutemi": "myocardialinfarction",
    "nfkb": "nfkappab",
    "nfκb": "nfkappab",
    "tak1": "map3k7",
    "mekk3": "map3k3",
    "il6": "il6",
}


def normalize(e: str) -> str:
    n = re.sub(r"[^a-z0-9]", "", e.lower())
    return SYNONYMS.get(n, n)


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
    return re.sub(r"[0-9]+$", "", re.sub(r"[^a-z0-9]", "", sym.lower()))


def is_near_neighbour(a: str, b: str) -> bool:
    """True when two *different* symbols are easily confused: same alphabetic stem with a small
    edit distance (MAP3K3/MAP3K7, MAPK3), or two drugs sharing a class suffix (-statin)."""
    na, nb = normalize(a), normalize(b)
    if na == nb or not na or not nb:
        return False
    if _alpha_stem(na) and _alpha_stem(na) == _alpha_stem(nb) and _levenshtein(na, nb) <= 2:
        return True
    if _levenshtein(na, nb) <= 1 and max(len(na), len(nb)) >= 4:
        return True
    ma, mb = _DRUG_SUFFIX.match(na), _DRUG_SUFFIX.match(nb)
    return bool(ma and mb and ma.group("suffix") == mb.group("suffix"))
