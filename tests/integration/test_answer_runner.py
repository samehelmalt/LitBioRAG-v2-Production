import json

from app.generation.base import TemplateGenerator
from app.reranking.base import LexicalOverlapReranker
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.encoders import HashEncoder
from app.retrieval.pipeline import HybridRetriever, RetrievalConfig
from app.retrieval.store import MetadataStore
from app.verification.base import OverlapVerifier
from app.verification.entities import RegexBiomedEntityExtractor
from evaluation.datasets.loaders import QueryRecord
from evaluation.runners.answer import run_answers, write_run
from scripts import calibrate_gate
from tests.retrieval.fixture_corpus import PLANTED_QUERIES


def _retriever(bundle):
    store = MetadataStore(bundle / "metadata.sqlite", read_only=True)
    return HybridRetriever(store, BM25Index(bundle / "bm25"),
                           DenseIndex(bundle / "dense", HashEncoder(128)), LexicalOverlapReranker(),
                           RetrievalConfig(k_bm25=50, k_dense=50, rerank_pool=30))  # fmt: skip


def _records():
    for i, (q, gold, qt) in enumerate(PLANTED_QUERIES):
        yield QueryRecord(id=f"q{i}", question=q, gold_pmids=tuple(gold), dataset="fixture",
                          question_type=qt, yn_label="yes" if qt == "yesno" else "")  # fmt: skip
    yield QueryRecord(id="oos", question="Does aspirin prevent migraine in adolescents?",
                      gold_pmids=("1",), dataset="fixture", question_type="yesno",
                      yn_label="no")  # fmt: skip
    yield QueryRecord(id="scope", question="What dose of aspirin should I take for my headache?",
                      gold_pmids=("1",), dataset="fixture", question_type="factoid")  # fmt: skip


def _run(bundle):
    return run_answers(
        _retriever(bundle), _records(), generator=TemplateGenerator(),
        verifier=OverlapVerifier(), extractor=RegexBiomedEntityExtractor(),
    )  # fmt: skip


def test_answer_runner_end_to_end(fixture_bundle, tmp_path):
    rows = _run(fixture_bundle)
    by = {r.query_id: r for r in rows}
    assert by["scope"].action == "ESCALATE" and by["scope"].rule == "R1-scope"
    assert by["oos"].action == "ABSTAIN"  # template generator abstains: nothing overlaps
    answered = [r for r in rows if r.action in ("ANSWER", "ANSWER_WITH_CAUTION")]
    assert answered and all(r.gold_in_evidence for r in answered)
    assert all(not r.substitution_detected and r.fabricated == 0 for r in answered)
    s = write_run(rows, tmp_path / "run", meta={"x": 1})
    assert s["labels"] == "proxy" and "UNCALIBRATED" in s["calibrated_on"]
    assert s["gate"]["false_pass_rate"] in (None, 0.0)
    assert (tmp_path / "run" / "report.md").read_text().startswith("# Answer run (labels: proxy)")
    first = json.loads((tmp_path / "run" / "rows.jsonl").read_text().splitlines()[0])
    assert set(first["features"]) >= {"entailment_rate", "sufficiency"}


def test_calibrate_gate_refuses_small_sets_and_fits_when_enough(fixture_bundle, tmp_path):
    rows = _run(fixture_bundle)
    run_dir = tmp_path / "calib"
    write_run(rows, run_dir)
    out = tmp_path / "gate.yaml"
    out.write_text("costs: {false_pass: 10.0, false_block: 1.0, warn: 0.3, abstain: 1.0}\n")
    rc = calibrate_gate.main(["--rows", str(run_dir / "rows.jsonl"), "--out", str(out),
                              "--run-id", "t", "--min-rows", "50"])  # fmt: skip
    assert rc == 2  # six rows are not a calibration set
    rc = calibrate_gate.main(["--rows", str(run_dir / "rows.jsonl"), "--out", str(out),
                              "--run-id", "t", "--min-rows", "1"])  # fmt: skip
    assert rc == 0
    import yaml

    cfg = yaml.safe_load(out.read_text())
    assert cfg["calibrated_on"] == "t" and cfg["confidence_model"]["weights"]
    assert 0.0 <= cfg["confidence_answer"] <= 1.0
    assert (run_dir / "calibration.json").exists()
