"""The near-neighbour substitution suite must be caught 100% with 0 false alarms on the fixture."""

from app.core.documents import Document, SourceType
from app.verification.base import OverlapVerifier
from app.verification.entities import RegexBiomedEntityExtractor
from evaluation.adversarial.substitution import build_cases, run_cases, summarize, write_run
from scripts import run_adversarial
from tests.retrieval.fixture_corpus import fixture_documents


def extra_docs():
    yield Document(doc_id="pmid:910001", pmid="910001", year=2019,
                   title="Atorvastatin lowers LDL cholesterol in familial hypercholesterolaemia",
                   abstract="Atorvastatin 40 mg reduced LDL cholesterol by 39% at 12 weeks.",
                   source_type=SourceType.RCT)  # fmt: skip
    yield Document(doc_id="pmid:910002", pmid="910002", year=2018,
                   title="Simvastatin and muscle symptoms in older adults",
                   abstract="Simvastatin 20 mg was associated with myalgia in 8% of participants.",
                   source_type=SourceType.OBSERVATIONAL)  # fmt: skip
    yield Document(doc_id="pmid:910003", pmid="910003", year=2020,
                   title="MAPK3 phosphorylation in cardiac hypertrophy",
                   abstract="MAPK3 (ERK1) phosphorylation increased threefold after overload.",
                   source_type=SourceType.JOURNAL_ARTICLE)  # fmt: skip


def test_fixture_suite_is_fully_caught_without_false_alarms(tmp_path):
    docs = list(fixture_documents()) + list(extra_docs())
    cases = build_cases(docs)
    pairs = {(c.asked_symbol, c.answer_symbol) for c in cases if c.kind == "attack"}
    assert ("MAP3K3", "MAP3K7") in pairs and ("MAP3K7", "MAP3K3") in pairs
    assert ("MAP3K3", "MAPK3") in pairs
    assert len(cases) >= 8 and len([c for c in cases if c.kind == "control"]) == len(cases) // 2

    results = run_cases(cases, {d.doc_id: d for d in docs}, OverlapVerifier(),
                        RegexBiomedEntityExtractor())  # fmt: skip
    s = summarize(results)
    assert s["catch_rate"] == 1.0, s["missed"]
    assert s["false_alarm_rate"] == 0.0, s["false_alarms"]
    written = write_run(cases, results, tmp_path / "adv")
    assert written["n_attacks"] == written["n_controls"]
    assert (tmp_path / "adv" / "cases.jsonl").exists()


def test_cli_gates_on_catch_rate(fixture_bundle, tmp_path):
    rc = run_adversarial.main(["--bundle", str(fixture_bundle), "--profile", "test",
                               "--out", str(tmp_path / "adv")])  # fmt: skip
    assert rc == 0
