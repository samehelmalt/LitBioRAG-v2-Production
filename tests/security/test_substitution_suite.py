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
    yield Document(doc_id="pmid:910004", pmid="910004", year=2021,
                   title="Hypokalemia and QT prolongation in heart failure",
                   abstract="Hypokalemia prolonged the QT interval in 23% of admissions.",
                   source_type=SourceType.OBSERVATIONAL)  # fmt: skip
    yield Document(doc_id="pmid:910005", pmid="910005", year=2021,
                   title="Hyperkalemia management in chronic kidney disease",
                   abstract="Hyperkalemia was treated with patiromer in 61% of patients.",
                   source_type=SourceType.OBSERVATIONAL)  # fmt: skip
    yield Document(doc_id="pmid:910006", pmid="910006", year=2022,
                   title="Type 2 diabetes remission after bariatric surgery",
                   abstract="Type 2 diabetes remitted in 54% of patients at two years.",
                   source_type=SourceType.RCT)  # fmt: skip
    yield Document(doc_id="pmid:910007", pmid="910007", year=2022,
                   title="Type 1 diabetes and closed-loop insulin delivery",
                   abstract="Type 1 diabetes time-in-range improved by 11 percentage points.",
                   source_type=SourceType.RCT)  # fmt: skip
    yield Document(doc_id="pmid:910008", pmid="910008", year=2019,
                   title="Hepatitis B vaccination coverage in newborns",
                   abstract="Hepatitis B birth-dose coverage reached 87% nationally.",
                   source_type=SourceType.OBSERVATIONAL)  # fmt: skip
    yield Document(doc_id="pmid:910009", pmid="910009", year=2019,
                   title="Hepatitis C cure rates with direct-acting antivirals",
                   abstract="Hepatitis C sustained virologic response was 96% at 12 weeks.",
                   source_type=SourceType.RCT)  # fmt: skip


def test_fixture_suite_is_fully_caught_without_false_alarms(tmp_path):
    docs = list(fixture_documents()) + list(extra_docs())
    x = RegexBiomedEntityExtractor()
    cases = build_cases(docs, x)
    pairs = {(c.asked_symbol.lower(), c.answer_symbol.lower()) for c in cases if c.kind == "attack"}
    assert ("map3k3", "map3k7") in pairs and ("map3k7", "map3k3") in pairs
    assert ("map3k3", "mapk3") in pairs
    assert ("atorvastatin", "simvastatin") in pairs
    assert ("hypokalemia", "hyperkalemia") in pairs
    assert ("type 2 diabetes", "type 1 diabetes") in pairs
    assert ("hepatitis b", "hepatitis c") in pairs
    reasons = {c.reason for c in cases}
    assert {"numbered_family", "drug_class:statin", "antonym_prefix", "letter_variant"} <= reasons
    assert len(cases) >= 20 and len([c for c in cases if c.kind == "control"]) == len(cases) // 2

    results = run_cases(cases, {d.doc_id: d for d in docs}, OverlapVerifier(), x)
    s = summarize(results)
    assert s["catch_rate"] == 1.0, s["missed"]
    assert s["false_alarm_rate"] == 0.0, s["false_alarms"]
    assert set(s["by_kind"]) >= {"gene", "drug", "condition"}
    written = write_run(cases, results, tmp_path / "adv")
    assert written["n_attacks"] == written["n_controls"]
    assert (tmp_path / "adv" / "cases.jsonl").exists()


def test_cli_gates_on_catch_rate(fixture_bundle, tmp_path):
    rc = run_adversarial.main(["--bundle", str(fixture_bundle), "--profile", "test",
                               "--out", str(tmp_path / "adv")])  # fmt: skip
    assert rc == 0
