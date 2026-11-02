"""Lecture QA runner: the prototype acceptance evaluation for the education product.

Runs every question of a lecture question set through ``AnswerPipeline`` (follow-ups in chain
with the delivered answer of their parent as context) and reports, per question and overall:

* slide retrieval: Recall@5/10, MRR over gold slides (canonical course/lecture/slide keys);
* citation-resolution validity: every delivered citation resolves to a slide of the deck and the
  cited span lies inside that slide's text (target 100%);
* answer correctness proxy: gold-answer token coverage in the delivered answer (a human label
  column is written next to it for review);
* delivered-answer precision (proxy), coverage, false PASS (delivered but wrong/uncited),
  abstention quality on unanswerable questions, repetition/format failure rates, latency, cost.

Numbers measured with test doubles are pipeline checks, not accuracy; with real models they
become the prototype report (docs/QUALITY_PLAYBOOK.md). Nothing here is read at runtime.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from evaluation.datasets.lecture_qa import LectureQuestion, slide_key
from evaluation.metrics.retrieval import mrr, recall_at_k

_WORD = re.compile(r"[a-z0-9.]+")
_STOP = {"the", "a", "an", "of", "with", "and", "or", "to", "in", "is", "are", "per", "for", "by",
         "it", "that", "this", "then", "first"}


def _tokens(s: str) -> set[str]:
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP and len(w) > 1}


@dataclass
class LectureRow:
    id: str
    type: str
    question: str
    resolved_question: str
    answerable: bool
    gold_slides: list[str]
    retrieved_slides: list[str]
    recall_at_5: float | None
    recall_at_10: float | None
    mrr: float | None
    action: str
    rule: str
    released: bool
    answer: str
    cited_slides: list[str]
    citations_valid: bool
    citation_errors: list[str]
    gold_coverage: float | None
    correct_proxy: bool | None
    human_label: str = ""  # filled by reviewers: correct | incorrect | partial
    n_attempts: int = 1
    quality_defects: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    usage: dict | None = None


def _slide_keys_from_sources(sources) -> list[str]:
    keys: list[str] = []
    for s in sources:
        cid = getattr(s, "canonical_id", None)
        if cid:
            k = slide_key(cid)
            if k not in keys:
                keys.append(k)
    return keys


def _citation_check(out, slide_lookup) -> tuple[list[str], bool, list[str]]:
    """Delivered citations must resolve to deck slides and the cited span must sit inside the
    slide text. ``slide_lookup(citation_id) -> (slide_key, text) | None``."""
    cited: list[str] = []
    errors: list[str] = []
    by_pid = {u.passage_id: u for u in out.evidence}
    removed = set(out.decision.removed_claims)
    for v in out.report.verdicts:
        if v.text in removed:
            continue
        for cid in v.cited_ids:
            res = slide_lookup(cid)
            if res is None:
                errors.append(f"{v.claim_id}: {cid} does not resolve to a slide")
                continue
            key, text = res
            if key not in cited:
                cited.append(key)
            u = by_pid.get(v.best_passage_id or "")
            if u is not None and u.citation_id == cid and u.text.strip() not in text:
                errors.append(f"{v.claim_id}: cited span not found in {cid}")
    return cited, not errors, errors


def run_lecture_qa(
    pipeline,
    questions: Sequence[LectureQuestion],
    *,
    kb_ids: list[str],
    slide_lookup,
    scope: str = "uploaded",
) -> list[LectureRow]:
    rows: list[LectureRow] = []
    delivered: dict[str, dict] = {}  # question id -> {"question", "answer"} for follow-ups
    for q in questions:
        history: list[str] = []
        if q.follow_up_of and q.follow_up_of in delivered:
            d = delivered[q.follow_up_of]
            history = [f"Q: {d['question']} → delivered answer: {d['answer']}"]
        t0 = time.perf_counter()
        out = pipeline.run(q.question, scope=scope, kb_ids=kb_ids, history=history,
                           request_id=f"lqa-{q.id}")  # fmt: skip
        latency = (time.perf_counter() - t0) * 1000
        retrieved = _slide_keys_from_sources(out.evidence)
        gold = set(q.gold_slides)
        released = out.decision.action.value in ("ANSWER", "ANSWER_WITH_CAUTION")
        cited, valid, errs = _citation_check(out, slide_lookup)
        answer = out.decision.answer_text if released else ""
        cov = None
        correct = None
        if q.answerable and q.gold_answer:
            g = _tokens(q.gold_answer)
            cov = len(g & _tokens(answer)) / len(g) if g and answer else 0.0
            correct = released and cov >= 0.5 and bool(gold & set(cited))
        if released:
            delivered[q.id] = {"question": q.question, "answer": answer}
        rows.append(LectureRow(
            id=q.id, type=q.type, question=q.question, resolved_question=out.question,
            answerable=q.answerable, gold_slides=sorted(gold), retrieved_slides=retrieved,
            recall_at_5=recall_at_k(retrieved, gold, 5) if gold else None,
            recall_at_10=recall_at_k(retrieved, gold, 10) if gold else None,
            mrr=mrr(retrieved, gold) if gold else None,
            action=out.decision.action.value, rule=out.decision.rule, released=released,
            answer=answer[:600], cited_slides=cited, citations_valid=valid,
            citation_errors=errs, gold_coverage=cov, correct_proxy=correct,
            n_attempts=len(out.attempts),
            quality_defects=[d for a in out.attempts for d in a.quality.defects],
            latency_ms=round(latency, 1), usage=out.generated.usage,
        ))  # fmt: skip
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def summarize(rows: Sequence[LectureRow]) -> dict:
    answerable = [r for r in rows if r.answerable]
    unans = [r for r in rows if not r.answerable]
    released = [r for r in rows if r.released]
    rel_ans = [r for r in released if r.answerable]
    by_type: dict[str, dict] = {}
    for t in sorted({r.type for r in rows}):
        rs = [r for r in rows if r.type == t]
        by_type[t] = {"n": len(rs), "released": sum(r.released for r in rs),
                      "correct_proxy": _mean([r.correct_proxy for r in rs if r.answerable]),
                      "recall_at_5": _mean([r.recall_at_5 for r in rs])}  # fmt: skip
    return {
        "n": len(rows),
        "labels": "proxy (gold-answer coverage); human_label column empty until reviewed",
        "retrieval": {"recall_at_5": _mean([r.recall_at_5 for r in answerable]),
                      "recall_at_10": _mean([r.recall_at_10 for r in answerable]),
                      "mrr": _mean([r.mrr for r in answerable])},
        "citations": {"validity": _mean([r.citations_valid for r in released]),
                      "errors": sum(len(r.citation_errors) for r in released)},
        "answers": {
            "coverage": len(released) / len(rows) if rows else None,
            "delivered_precision_proxy": _mean([r.correct_proxy for r in rel_ans]),
            "correct_rate_proxy": _mean([r.correct_proxy for r in answerable]),
            "false_pass_proxy": _mean([(not r.correct_proxy) for r in rel_ans]),
            "unanswerable_abstained": _mean([(not r.released) for r in unans]),
            "unanswerable_false_answers": sum(r.released for r in unans),
        },
        "generation": {"mean_attempts": _mean([r.n_attempts for r in rows]),
                       "defect_rate": _mean([bool(r.quality_defects) for r in rows])},
        "latency_ms": {"p50": _mean([r.latency_ms for r in rows]),
                       "max": max((r.latency_ms for r in rows), default=None)},
        "cost_usd": _mean([(r.usage or {}).get("cost_usd") for r in rows]),
        "by_type": by_type,
        "targets": {"citation_validity": 1.0, "unauthorized_disclosures": 0,
                    "delivered_precision": "owner-chosen operating point (QUALITY_PLAYBOOK.md)"},
    }


def write_run(rows: Sequence[LectureRow], out_dir: Path, versions: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rows.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r), ensure_ascii=False, default=str) + "\n")
    s = summarize(rows)
    s["versions"] = versions or {}
    (out_dir / "summary.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    lines = ["# Lecture QA report", "", f"questions: {s['n']}", "",
             "| metric | value |", "|---|---|"]
    for grp in ("retrieval", "citations", "answers", "generation"):
        for k, v in s[grp].items():
            lines.append(f"| {grp}.{k} | {v} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    return s
