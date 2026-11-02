"""End-to-end answer runner: retrieve → evidence → generate → verify → gate, per query.

Rows carry the action, the deciding rule, every verification field, the proxy label and the
features the calibration job fits on. Trust metrics come from evaluation/metrics/trust.py with the
mapping ANSWER→PASS, ANSWER_WITH_CAUTION→WARN, CLARIFY/ABSTAIN/ESCALATE→BLOCK.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.evidence.selector import select_evidence
from app.generation.attempts import run_attempts
from app.generation.service import GenerationConfig
from app.generation.service import answer as generate_answer
from app.retrieval.pipeline import HybridRetriever
from app.safety.gate import GateConfig, confidence_features, decide
from app.safety.sufficiency import evidence_sufficiency, sufficiency_features
from app.verification.pipeline import VerificationConfig, verify
from evaluation.datasets.loaders import QueryRecord
from evaluation.labels import proxy_acceptable
from evaluation.metrics.performance import latency_summary
from evaluation.metrics.trust import brier, coverage_risk_curve, ece, gate_confusion
from evaluation.runners.retrieval import gold_doc_ids

_YN = re.compile(r"^\s*(yes|no|maybe)\b", re.IGNORECASE)


@dataclass
class AnswerRow:
    query_id: str
    question: str
    dataset: str
    question_type: str
    gold: list[str]
    gold_in_evidence: bool
    action: str
    internal_state: str
    rule: str
    reason: str
    confidence: float | None
    sufficiency: float | None
    abstained: bool
    n_claims: int
    entailment_rate: float | None
    unsupported_rate: float | None
    contradiction_rate: float | None
    citation_support_rate: float | None
    fabricated: int
    substitution_detected: bool
    entity_fidelity: float | None
    numeric_fidelity: float | None
    n_conflicts: int
    yn_gold: str
    yn_pred: str
    proxy_acceptable: bool
    features: dict[str, float] = field(default_factory=dict)
    sufficiency_features: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    answer_text: str = ""
    # metric separation: candidate = attempt 1 raw output, verified = selected attempt,
    # released = delivered by the gate (ANSWER / ANSWER_WITH_CAUTION)
    n_attempts: int = 1
    candidate_quality_ok: bool = True
    candidate_entailment_rate: float | None = None
    candidate_vetoed: bool = False
    released: bool = False


def run_answers(
    retriever: HybridRetriever,
    records: Iterable[QueryRecord],
    *,
    generator,
    verifier,
    extractor,
    gen_cfg: GenerationConfig | None = None,
    ver_cfg: VerificationConfig | None = None,
    gate_cfg: GateConfig | None = None,
    thresholds: dict | None = None,
    max_units: int = 8,
) -> list[AnswerRow]:
    gate_cfg = gate_cfg or GateConfig()
    rows: list[AnswerRow] = []
    for rec in records:
        t0 = time.perf_counter()
        res = retriever.retrieve(rec.question, mode="hybrid+rerank")
        units = select_evidence(res, retriever.store, max_units=max_units)
        def do_generate(feedback, _q=rec.question, _u=units):
            parsed = generate_answer(_q, _u, generator, config=gen_cfg, feedback=feedback)
            return parsed.raw, parsed

        def do_verify(parsed, _q=rec.question, _u=units):
            return verify(parsed, _u, verifier, extractor, config=ver_cfg, thresholds=thresholds,
                          question=_q)  # fmt: skip

        attempts, selected = run_attempts(
            do_generate, do_verify, max_attempts=(gen_cfg or GenerationConfig()).max_attempts,
            question_entities=extractor.extract(rec.question) or None,
            extract_entities=extractor.extract, units=units,
        )
        gen = selected.answer
        rep = selected.verification if selected.verification is not None else do_verify(gen)
        first = attempts[0]
        suff = evidence_sufficiency(res, rep, gate_cfg.sufficiency_model)
        dec = decide(rep, gen, sufficiency=suff, config=gate_cfg, question=rec.question)
        gold = gold_doc_ids(rec)
        gold_in_ev = any(u.doc_id in gold for u in units)
        m = _YN.match(gen.answer or "")
        yn_pred = m.group(1).lower() if m else ""
        rows.append(
            AnswerRow(
                query_id=rec.id, question=rec.question, dataset=rec.dataset,
                question_type=rec.question_type, gold=sorted(gold), gold_in_evidence=gold_in_ev,
                action=dec.action.value, internal_state=dec.internal_state, rule=dec.rule,
                reason=dec.reason, confidence=dec.confidence, sufficiency=suff,
                abstained=gen.abstained, n_claims=rep.n_claims,
                entailment_rate=rep.entailment_rate, unsupported_rate=rep.unsupported_rate,
                contradiction_rate=rep.contradiction_rate,
                citation_support_rate=rep.citation_support_rate,
                fabricated=len(rep.fabricated_citations),
                substitution_detected=rep.substitution_detected,
                entity_fidelity=rep.entity_fidelity, numeric_fidelity=rep.numeric_fidelity,
                n_conflicts=len(rep.conflicts), yn_gold=rec.yn_label, yn_pred=yn_pred,
                n_attempts=len([a for a in attempts if a.raw]),
                candidate_quality_ok=first.quality.ok,
                candidate_entailment_rate=(first.verification.entailment_rate
                                           if first.verification else None),
                candidate_vetoed=first.vetoed,
                released=dec.action.value in ("ANSWER", "ANSWER_WITH_CAUTION"),
                proxy_acceptable=proxy_acceptable(
                    yn_gold=rec.yn_label or None, yn_pred=yn_pred or None,
                    gold_retrieved=gold_in_ev, unsupported_rate=rep.unsupported_rate,
                    contradiction_rate=rep.contradiction_rate,
                ) and not rep.substitution_detected and not rep.fabricated_citations,
                features=confidence_features(rep, suff),
                sufficiency_features=sufficiency_features(res, rep),
                latency_ms=(time.perf_counter() - t0) * 1000,
                answer_text=dec.answer_text[:500],
            )  # fmt: skip
        )
    return rows


def _mean(xs) -> float | None:
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else None


def summarize(rows: Sequence[AnswerRow], calibrated_on: str = "") -> dict:
    decisions = [r.internal_state for r in rows]
    labels = [r.proxy_acceptable for r in rows]
    released = [r for r in rows if r.released]
    out: dict = {
        "n": len(rows),
        "labels": "proxy" if rows else None,
        "channels": {  # the same metric at three stages; they are not interchangeable
            "candidate": {
                "quality_ok_rate": _mean([r.candidate_quality_ok for r in rows]),
                "vetoed_rate": _mean([r.candidate_vetoed for r in rows]),
                "entailment_rate": _mean([r.candidate_entailment_rate for r in rows
                                          if r.candidate_entailment_rate is not None]),
            },
            "verified": {
                "entailment_rate": _mean([r.entailment_rate for r in rows
                                          if r.entailment_rate is not None]),
                "mean_attempts": _mean([r.n_attempts for r in rows]),
            },
            "released": {
                "n": len(released),
                "coverage": len(released) / len(rows) if rows else None,
                "precision_proxy": _mean([r.proxy_acceptable for r in released]),
                "entailment_rate": _mean([r.entailment_rate for r in released
                                          if r.entailment_rate is not None]),
            },
        },
        "calibrated_on": calibrated_on or "UNCALIBRATED (heuristic defaults)",
        "actions": dict(Counter(r.action for r in rows)),
        "rules": dict(Counter(r.rule for r in rows)),
        "proxy_acceptable_rate": statistics.fmean(labels) if labels else None,
    }
    if rows:
        g = gate_confusion(decisions, labels)
        out["gate"] = g.__dict__ | {"counts": g.counts}
        conf = [r.confidence if r.confidence is not None else 0.0 for r in rows]
        out["ece"] = ece(conf, labels)
        out["brier"] = brier(conf, labels)
        out["coverage_risk"] = coverage_risk_curve(conf, labels)
        out["latency_ms"] = asdict(latency_summary([r.latency_ms for r in rows]))
    return out


def write_run(rows: Sequence[AnswerRow], out_dir: Path, *, meta: dict | None = None,
              calibrated_on: str = "") -> dict:  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rows.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r), ensure_ascii=False, default=str) + "\n")
    summary = summarize(rows, calibrated_on)
    summary["meta"] = meta or {}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    (out_dir / "report.md").write_text(render_report(summary))
    return summary


def render_report(s: dict) -> str:
    lines = ["# Answer run (labels: proxy)", "", f"Calibration: {s['calibrated_on']}", ""]
    lines += [f"Queries: {s['n']}", "", "| action | n |", "|---|---|"]
    for a, n in sorted(s["actions"].items()):
        lines.append(f"| {a} | {n} |")
    if "gate" in s:
        g = s["gate"]
        lines += ["", "| trust metric (proxy labels) | value |", "|---|---|"]
        for k in ("false_pass_rate", "false_warn_rate", "false_block_rate", "coverage",
                  "abstention_precision", "abstention_recall"):  # fmt: skip
            v = g.get(k)
            lines.append(f"| {k} | {'—' if v is None else f'{v:.3f}'} |")
        lines.append(f"| ECE | {s['ece']:.3f} |")
        lines.append(f"| Brier | {s['brier']:.3f} |")
    lines += ["", "| deciding rule | n |", "|---|---|"]
    for r, n in sorted(s["rules"].items()):
        lines.append(f"| {r} | {n} |")
    return "\n".join(lines) + "\n"
