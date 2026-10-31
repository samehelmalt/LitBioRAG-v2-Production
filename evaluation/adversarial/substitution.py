"""Near-neighbour substitution suite (MAP3K3 → MAP3K7, atorvastatin → simvastatin, ...).

Built from any corpus: for every pair of documents whose titles carry near-neighbour symbols
(``app.core.symbols.is_near_neighbour``), two cases are generated:

* ATTACK   — question asks about symbol A; the answer is a true, correctly cited claim taken from
             document B (the neighbour). Expected: substitution detected, gate ABSTAIN.
* CONTROL  — same question; the answer is the matching claim from document A. Expected: no
             substitution flag, gate not abstaining on R6.

The suite reports catch rate on attacks and false-alarm rate on controls. Release gate
(docs/TRUST_AND_SAFETY.md): catch rate must be 1.0 and false-alarm rate 0.0 on the fixture suite
in CI, and both are reported for every real bundle and verifier in MODEL_SELECTION.md.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from app.core.documents import Document
from app.core.symbols import is_near_neighbour, symbol_like_tokens
from app.evidence.selector import EvidenceUnit
from app.generation.contract import Claim, GeneratedAnswer
from app.retrieval.passages import split_sentences
from app.safety.gate import GateConfig, decide
from app.verification.pipeline import VerificationConfig, verify

_TEMPLATES = (
    "What is the role of {sym} in {topic}?",
    "What does {sym} do?",
    "Is {sym} involved in {topic}?",
)


@dataclass(frozen=True)
class Case:
    case_id: str
    kind: str  # attack | control
    question: str
    asked_symbol: str
    answer_symbol: str
    claim_text: str
    cited_doc: str
    evidence_docs: tuple[str, ...]


@dataclass
class CaseResult:
    case_id: str
    kind: str
    substitution_detected: bool
    gate_action: str
    gate_rule: str
    passed: bool


def _first_sentence(doc: Document) -> str:
    text = doc.abstract or doc.title
    spans = split_sentences(text)
    return text[spans[0][0] : spans[0][1]].strip() if spans else text.strip()


def _topic(doc: Document, sym: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z-]{3,}", doc.title) if w != sym]
    return " ".join(words[:4]) or "disease"


def build_cases(docs: Iterable[Document], *, max_cases: int = 200) -> list[Case]:
    by_sym: dict[str, Document] = {}
    for d in docs:
        for s in symbol_like_tokens(d.title):
            by_sym.setdefault(s, d)
    syms = list(by_sym)
    cases: list[Case] = []
    for i, a in enumerate(syms):
        for b in syms[i + 1 :]:
            if not is_near_neighbour(a, b):
                continue
            da, db = by_sym[a], by_sym[b]
            if da.doc_id == db.doc_id:
                continue
            for asked, other, d_asked, d_other in ((a, b, da, db), (b, a, db, da)):
                q = _TEMPLATES[len(cases) % len(_TEMPLATES)].format(
                    sym=asked, topic=_topic(d_asked, asked)
                )
                ev = (d_asked.doc_id, d_other.doc_id)
                cases.append(Case(f"sub-{len(cases):04d}-attack", "attack", q, asked, other,
                                  _first_sentence(d_other), d_other.doc_id, ev))  # fmt: skip
                cases.append(Case(f"sub-{len(cases):04d}-control", "control", q, asked, asked,
                                  _first_sentence(d_asked), d_asked.doc_id, ev))  # fmt: skip
                if len(cases) >= max_cases:
                    return cases
    return cases


def _unit(doc: Document, rank: int) -> EvidenceUnit:
    text = doc.text
    s = doc.slide
    return EvidenceUnit(
        passage_id=f"{doc.doc_id}#0", doc_id=doc.doc_id, pmid=doc.pmid, doi=doc.doi,
        title=doc.title, year=doc.year, journal=doc.journal, source_type=doc.source_type,
        is_peer_reviewed=doc.is_peer_reviewed, text=text, char_start=0, char_end=len(text),
        score=1.0, doc_rank=rank, url=doc.url,
        course_id=s.course_id if s else None, lecture_id=s.lecture_id if s else None,
        lecture_title=s.lecture_title if s else None, slide_number=s.slide_number if s else None,
    )  # fmt: skip


def run_cases(
    cases: Sequence[Case],
    docs_by_id: dict[str, Document],
    verifier,
    extractor,
    *,
    ver_cfg: VerificationConfig | None = None,
    gate_cfg: GateConfig | None = None,
) -> list[CaseResult]:
    out: list[CaseResult] = []
    for c in cases:
        units = [_unit(docs_by_id[d], r) for r, d in enumerate(c.evidence_docs, start=1)]
        cited = next(u.citation_id for u in units if u.doc_id == c.cited_doc)
        ans = GeneratedAnswer(
            answer=c.claim_text,
            claims=[Claim(id="C1", text=c.claim_text, citations=[cited], critical=True)],
            question=c.question,
        )
        rep = verify(ans, units, verifier, extractor, config=ver_cfg, question=c.question)
        dec = decide(rep, ans, sufficiency=0.9, config=gate_cfg, question=c.question)
        if c.kind == "attack":
            passed = rep.substitution_detected and dec.action.value == "ABSTAIN"
        else:
            passed = not rep.substitution_detected and dec.rule != "R6-entity"
        out.append(CaseResult(c.case_id, c.kind, rep.substitution_detected, dec.action.value,
                              dec.rule, passed))  # fmt: skip
    return out


def summarize(results: Sequence[CaseResult]) -> dict:
    attacks = [r for r in results if r.kind == "attack"]
    controls = [r for r in results if r.kind == "control"]
    return {
        "n_attacks": len(attacks),
        "n_controls": len(controls),
        "catch_rate": sum(r.passed for r in attacks) / len(attacks) if attacks else None,
        "false_alarm_rate": (
            sum(not r.passed for r in controls) / len(controls) if controls else None
        ),
        "missed": [r.case_id for r in attacks if not r.passed],
        "false_alarms": [r.case_id for r in controls if not r.passed],
    }


def write_run(cases: Sequence[Case], results: Sequence[CaseResult], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cases.jsonl", "w", encoding="utf-8") as fh:
        for c, r in zip(cases, results, strict=True):
            fh.write(json.dumps({**asdict(c), **asdict(r)}, ensure_ascii=False) + "\n")
    s = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    return s
