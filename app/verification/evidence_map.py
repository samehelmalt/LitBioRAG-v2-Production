"""Map each claim to the evidence passages it should be checked against, and score them.

Order: passages of the cited sources first (the claim says they support it), then the best
lexical matches among the remaining evidence up to ``max_pairs``. Every pair is scored by the
verifier; the per-passage scores are kept so that support and contradiction can be attributed to
specific passages and conflicts can be detected.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.evidence.selector import EvidenceUnit
from app.generation.contract import Claim
from app.verification.base import NLIScores, Verifier

_TOK = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class PassageScore:
    unit: EvidenceUnit
    scores: NLIScores
    cited: bool


def _overlap(a: str, b: str) -> int:
    return len(set(_TOK.findall(a.lower())) & set(_TOK.findall(b.lower())))


def candidates_for(
    claim: Claim, units: Sequence[EvidenceUnit], *, max_pairs: int
) -> list[tuple[EvidenceUnit, bool]]:
    cited = set(claim.citations)
    first = [u for u in units if u.citation_id in cited]
    rest = sorted(
        (u for u in units if u.citation_id not in cited),
        key=lambda u: -_overlap(claim.text, u.text),
    )
    out = [(u, True) for u in first] + [(u, False) for u in rest]
    return out[:max_pairs]


def score_claims(
    claims: Sequence[Claim],
    units: Sequence[EvidenceUnit],
    verifier: Verifier,
    *,
    max_pairs: int,
) -> dict[str, list[PassageScore]]:
    """One verifier call for all (passage, claim) pairs; results grouped per claim id."""
    plan: list[tuple[str, EvidenceUnit, bool]] = []
    for c in claims:
        for u, cited in candidates_for(c, units, max_pairs=max_pairs):
            plan.append((c.id, u, cited))
    texts = {c.id: c.text for c in claims}
    scores = verifier.score_pairs([(u.text, texts[cid]) for cid, u, _ in plan]) if plan else []
    out: dict[str, list[PassageScore]] = {c.id: [] for c in claims}
    for (cid, u, cited), s in zip(plan, scores, strict=True):
        out[cid].append(PassageScore(u, s, cited))
    return out
