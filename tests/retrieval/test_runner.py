import json

from app.reranking.base import LexicalOverlapReranker
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.encoders import HashEncoder
from app.retrieval.pipeline import HybridRetriever, RetrievalConfig
from app.retrieval.store import MetadataStore
from evaluation.datasets.loaders import QueryRecord
from evaluation.runners.retrieval import QueryRow, classify_failure, run_retrieval, write_run
from scripts import run_retrieval_baseline
from tests.retrieval.fixture_corpus import PLANTED_QUERIES


def _records():
    for i, (q, gold, qt) in enumerate(PLANTED_QUERIES):
        yield QueryRecord(id=f"q{i}", question=q, gold_pmids=tuple(gold), dataset="fixture",
                          question_type=qt)  # fmt: skip
    yield QueryRecord(id="q-cov", question="Does aspirin prevent migraine?",
                      gold_pmids=("1",), dataset="fixture", question_type="yesno")  # fmt: skip


def test_runner_on_fixture(fixture_bundle, tmp_path):
    store = MetadataStore(fixture_bundle / "metadata.sqlite", read_only=True)
    retriever = HybridRetriever(
        store, BM25Index(fixture_bundle / "bm25"),
        DenseIndex(fixture_bundle / "dense", HashEncoder(128)),
        LexicalOverlapReranker(), RetrievalConfig(k_bm25=50, k_dense=50, rerank_pool=30),
    )  # fmt: skip
    corpus = run_retrieval_baseline.corpus_doc_ids(store)
    rows = run_retrieval(retriever, _records(), mode="hybrid+rerank", corpus_doc_ids=corpus)
    assert len(rows) == 5
    planted = [r for r in rows if r.query_id != "q-cov"]
    assert all(r.recall["1"] == 1.0 and r.failure == "OK" for r in planted)
    assert all(r.gold_stage_ranks["final"] == 1 and r.in_rerank_pool for r in planted)
    cov = next(r for r in rows if r.query_id == "q-cov")
    assert cov.failure == "COV" and cov.n_gold_in_corpus == 0 and cov.recall["5"] is None

    summary = write_run(rows, tmp_path / "run", meta={"encoder": "hash:128"})
    assert summary["overall"]["recall@5"] == 1.0 and summary["overall"]["n_evaluable"] == 4
    assert summary["overall"]["failures"] == {"OK": 4, "COV": 1}
    assert set(summary["by_question_type"]) == {"factoid", "causal", "yesno"}
    assert (tmp_path / "run" / "rows.jsonl").exists()
    report = (tmp_path / "run" / "report.md").read_text()
    assert "recall@5 | 1.000" in report and "| COV | 1 |" in report
    first = json.loads((tmp_path / "run" / "rows.jsonl").read_text().splitlines()[0])
    assert first["gold_stage_ranks"]["bm25"] == 1


def _row(mode, ranks, frr=None, in_pool=True, n_in_corpus=1):
    return QueryRow("q", "q?", "d", "t", mode, ["pmid:1"], 1, n_in_corpus, {"20": 0.0}, None,
                    None, frr, ranks, in_pool, 1.0, "")  # fmt: skip


def test_failure_classification_rules():
    kw = dict(k=20, pool_k=100)
    base = {s: None for s in ("bm25", "dense", "fused", "reranked", "final")}
    assert classify_failure(_row("hybrid+rerank", base, frr=3), **kw) == "OK"
    assert classify_failure(_row("hybrid+rerank", base, n_in_corpus=0), **kw) == "COV"
    r = dict(base, bm25=2, dense=5, fused=4, reranked=57)
    assert classify_failure(_row("hybrid+rerank", r), **kw) == "RERANK"
    r = dict(base, bm25=3, dense=None, fused=45, reranked=70)
    assert classify_failure(_row("hybrid+rerank", r), **kw) == "RRF"
    r = dict(base, bm25=None, dense=7, fused=30)
    assert classify_failure(_row("hybrid", r), **kw) == "RRF"
    r = dict(base, bm25=None, dense=90, fused=150)
    assert classify_failure(_row("hybrid", r, in_pool=False), **kw) == "BM25"
    r = dict(base, bm25=150, dense=None, fused=190)
    assert classify_failure(_row("hybrid", r, in_pool=False), **kw) == "DENSE"
    r = dict(base, bm25=150, dense=120, fused=160)
    assert classify_failure(_row("hybrid", r, in_pool=False), **kw) == "POOL"
    only_bm25 = _row("bm25", dict(base, bm25=150), in_pool=False)
    assert classify_failure(only_bm25, **kw) == "POOL"
    assert classify_failure(_row("bm25", base), **kw) == "MISS"
    assert classify_failure(_row("hybrid+rerank", base), **kw) == "MISS"


def test_cli_end_to_end(fixture_bundle, tmp_path, monkeypatch):
    # thesis24 queries are not in the fixture corpus: every row must be COV, and the CLI must run
    out = tmp_path / "cli"
    rc = run_retrieval_baseline.main(
        ["--bundle", str(fixture_bundle), "--encoder", "hash:128", "--reranker", "lexical",
         "--queries", "thesis24", "--modes", "bm25", "hybrid+rerank", "--out", str(out),
         "--limit", "3"]  # fmt: skip
    )
    assert rc == 0
    s = json.loads((out / "hybrid_rerank" / "summary.json").read_text())
    assert s["overall"]["n_queries"] == 3 and s["overall"]["failures"] == {"COV": 3}
    assert s["meta"]["bundle"] == "fixture_v1" and s["meta"]["reranker"] == "lexical"
