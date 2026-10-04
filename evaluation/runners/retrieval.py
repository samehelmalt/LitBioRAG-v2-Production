"""Run a retriever over query records and report ranking metrics plus failure attribution.

Per query the runner records recall@k, MRR, nDCG@10, first relevant rank, whether a gold document
reached the reranker pool, the rank of the best gold document at every stage (bm25, dense, fused,
reranked, final), latency, and a failure code from the taxonomy in RETRIEVAL_FAILURE_ANALYSIS.md
where it can be decided mechanically:

    OK        a gold document is within the top K
    COV       no gold document exists in the corpus (not a retrieval failure)
    RERANK    gold was within K after fusion and the reranker pushed it out
    RRF       gold was within K in BM25 or dense but fusion pushed it out
    BM25      the BM25 list never retrieved gold; only dense did (lexical side missed it)
    DENSE     the dense list never retrieved gold; only BM25 did (dense side missed it)
    POOL      both first-stage lists retrieved gold but only deep (below K); depth/pool problem
    MISS      no stage retrieved gold; needs manual classification (QF / TERM / ACR / PRE / IDX)
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.retrieval.pipeline import HybridRetriever, RetrievalResult
from evaluation.datasets.loaders import QueryRecord
from evaluation.metrics.performance import latency_summary
from evaluation.metrics.retrieval import ranking_metrics

STAGES = ("bm25", "dense", "fused", "reranked", "final")


def gold_doc_ids(record: QueryRecord) -> set[str]:
    out = set()
    for g in record.gold_pmids:
        g = str(g)
        out.add(g if g.startswith(("pmid:", "doi:")) else f"pmid:{g}")
    return out


@dataclass
class QueryRow:
    query_id: str
    question: str
    dataset: str
    question_type: str
    mode: str
    gold: list[str]
    n_gold: int
    n_gold_in_corpus: int
    recall: dict[str, float | None]  # "1","5","10","20" -> corpus-normalized recall (None if n/a)
    mrr: float | None
    ndcg10: float | None
    first_relevant_rank: int | None
    gold_stage_ranks: dict[str, int | None]  # best gold doc rank per stage
    in_rerank_pool: bool
    latency_ms: float
    failure: str
    top5: list[str] = field(default_factory=list)


def best_gold_rank(result: RetrievalResult, gold: set[str], stage: str) -> int | None:
    ranks = [r for g in gold if (r := result.audit.rank_of(g, stage)) is not None]
    return min(ranks) if ranks else None


def classify_failure(row: QueryRow, *, k: int, pool_k: int) -> str:
    if row.n_gold_in_corpus == 0:
        return "COV"
    if row.first_relevant_rank is not None and row.first_relevant_rank <= k:
        return "OK"
    r = row.gold_stage_ranks
    within = {s: (r.get(s) is not None and r[s] <= k) for s in STAGES}
    found = {s: r.get(s) is not None for s in STAGES}
    if row.mode == "hybrid+rerank" and within["fused"] and not within["reranked"]:
        return "RERANK"
    if row.mode in ("hybrid", "hybrid+rerank"):
        if (within["bm25"] or within["dense"]) and not within["fused"]:
            return "RRF"
        if found["dense"] and not found["bm25"]:
            return "BM25"
        if found["bm25"] and not found["dense"]:
            return "DENSE"
        if found["bm25"] and found["dense"]:
            return "POOL"
        return "MISS"
    return "POOL" if found.get(row.mode) else "MISS"


def run_retrieval(
    retriever: HybridRetriever,
    records: Iterable[QueryRecord],
    *,
    mode: str,
    ks: Sequence[int] = (1, 5, 10, 20),
    corpus_doc_ids: set[str] | None = None,
) -> list[QueryRow]:
    k_max = max(ks)
    pool_k = retriever.cfg.rerank_pool
    rows: list[QueryRow] = []
    for rec in records:
        gold = gold_doc_ids(rec)
        if not gold:
            continue
        t = time.perf_counter()
        res = retriever.retrieve(rec.question, mode=mode)
        latency = (time.perf_counter() - t) * 1000
        ranked = [d.doc_id for d in res.docs]
        in_corpus = gold & corpus_doc_ids if corpus_doc_ids is not None else gold
        recall: dict[str, float | None] = {}
        mrr = ndcg = None
        if in_corpus:
            for k in ks:
                m = ranking_metrics(ranked, in_corpus, k=k)
                recall[str(k)] = m.recall_at_k
            m10 = ranking_metrics(ranked, in_corpus, k=10)
            mrr, ndcg = m10.mrr, m10.ndcg
        else:
            recall = {str(k): None for k in ks}
        frr = next((i for i, d in enumerate(ranked, start=1) if d in in_corpus), None)
        stage_ranks = {s: best_gold_rank(res, in_corpus, s) for s in STAGES}
        pool_stage = "fused" if mode.startswith("hybrid") else mode
        fused_rank = stage_ranks.get(pool_stage)
        row = QueryRow(
            query_id=rec.id,
            question=rec.question,
            dataset=rec.dataset,
            question_type=rec.question_type,
            mode=mode,
            gold=sorted(gold),
            n_gold=len(gold),
            n_gold_in_corpus=len(in_corpus),
            recall=recall,
            mrr=mrr,
            ndcg10=ndcg,
            first_relevant_rank=frr,
            gold_stage_ranks=stage_ranks,
            in_rerank_pool=fused_rank is not None and fused_rank <= pool_k,
            latency_ms=latency,
            failure="",
            top5=ranked[:5],
        )
        row.failure = classify_failure(row, k=k_max, pool_k=pool_k)
        rows.append(row)
    return rows


def _mean(vals: Iterable[float | None]) -> float | None:
    xs = [v for v in vals if v is not None]
    return statistics.fmean(xs) if xs else None


def summarize(rows: Sequence[QueryRow], ks: Sequence[int] = (1, 5, 10, 20)) -> dict:
    def block(rs: Sequence[QueryRow]) -> dict:
        evaluable = [r for r in rs if r.n_gold_in_corpus > 0]
        out = {
            "n_queries": len(rs),
            "n_evaluable": len(evaluable),
            "mrr": _mean(r.mrr for r in evaluable),
            "ndcg10": _mean(r.ndcg10 for r in evaluable),
            "first_relevant_rank_median": (
                statistics.median([r.first_relevant_rank for r in evaluable
                                   if r.first_relevant_rank is not None])  # fmt: skip
                if any(r.first_relevant_rank is not None for r in evaluable) else None
            ),
            "in_rerank_pool_rate": _mean(float(r.in_rerank_pool) for r in evaluable),
            "failures": dict(Counter(r.failure for r in rs)),
        }
        for k in ks:
            out[f"recall@{k}"] = _mean(r.recall.get(str(k)) for r in evaluable)
        return out

    by_dataset: dict[str, list[QueryRow]] = defaultdict(list)
    by_type: dict[str, list[QueryRow]] = defaultdict(list)
    for r in rows:
        by_dataset[r.dataset].append(r)
        by_type[r.question_type].append(r)
    lat = latency_summary([r.latency_ms for r in rows]) if rows else None
    return {
        "mode": rows[0].mode if rows else None,
        "overall": block(rows),
        "by_dataset": {k: block(v) for k, v in sorted(by_dataset.items())},
        "by_question_type": {k: block(v) for k, v in sorted(by_type.items())},
        "latency_ms": asdict(lat) if lat else None,
    }


def write_run(rows: Sequence[QueryRow], out_dir: Path, *, meta: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rows.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    summary = summarize(rows)
    summary["meta"] = meta or {}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out_dir / "report.md").write_text(render_report(summary, rows))
    return summary


def render_report(summary: dict, rows: Sequence[QueryRow]) -> str:
    o = summary["overall"]
    lines = [f"# Retrieval run — mode `{summary['mode']}`", ""]
    meta = summary.get("meta") or {}
    if meta:
        lines += ["```json", json.dumps(meta, indent=1), "```", ""]
    lines.append(
        f"Queries: {o['n_queries']} (evaluable: {o['n_evaluable']}). "
        f"Reranker-pool hit rate: {_fmt(o['in_rerank_pool_rate'])}."
    )
    lines += ["", "| metric | value |", "|---|---|"]
    for key in ("recall@1", "recall@5", "recall@10", "recall@20", "mrr", "ndcg10"):
        lines.append(f"| {key} | {_fmt(o.get(key))} |")
    lines.append(f"| first relevant rank (median) | {o['first_relevant_rank_median']} |")
    if summary.get("latency_ms"):
        lat = summary["latency_ms"]
        p = f"{lat['p50']:.0f} / {lat['p95']:.0f} / {lat['p99']:.0f}"
        lines.append(f"| latency ms p50 / p95 / p99 | {p} |")
    lines += ["", "## Failure codes", "", "| code | n |", "|---|---|"]
    for code, n in sorted(o["failures"].items()):
        lines.append(f"| {code} | {n} |")
    lines += ["", "## By question type", ""]
    lines += ["| type | n | recall@5 | recall@10 | mrr |", "|---|---|---|---|---|"]
    for t, b in summary["by_question_type"].items():
        cells = (b["n_queries"], _fmt(b.get("recall@5")), _fmt(b.get("recall@10")), _fmt(b["mrr"]))
        lines.append(f"| {t} | " + " | ".join(str(c) for c in cells) + " |")
    failed = [r for r in rows if r.failure not in ("OK", "COV")]
    if failed:
        lines += ["", "## Failed queries", ""]
        lines += ["| code | question | gold stage ranks |", "|---|---|---|"]
        for r in failed:
            present = [f"{s}={v}" for s, v in r.gold_stage_ranks.items() if v is not None]
            lines.append(f"| {r.failure} | {r.question[:90]} | {', '.join(present) or '—'} |")
    return "\n".join(lines) + "\n"


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"
