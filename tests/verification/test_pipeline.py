import pytest

from app.core.documents import SourceType
from app.evidence.selector import EvidenceUnit
from app.generation.base import TemplateGenerator
from app.generation.service import answer
from app.verification.base import OverlapVerifier
from app.verification.entities import RegexBiomedEntityExtractor
from app.verification.pipeline import VerificationConfig, verify


def unit(pid, text, **kw) -> EvidenceUnit:
    base = dict(passage_id=f"pmid:{pid}#0", doc_id=f"pmid:{pid}", pmid=str(pid), doi=None,
                title=f"Title {pid}", year=2016, journal="J",
                source_type=SourceType.JOURNAL_ARTICLE, is_peer_reviewed=True, text=text,
                char_start=0, char_end=len(text), score=1.0, doc_rank=1, url=None)  # fmt: skip
    base.update(kw)
    return EvidenceUnit(**base)


EVIDENCE = [
    unit(900001, "MAP3K3 (MEKK3) is required for NF-kappaB signalling. Knockdown of MAP3K3 "
                 "reduced p65 nuclear translocation by 61%."),
    unit(900002, "MAP3K7, also known as TAK1, mediates TLR signalling. This study does not "
                 "examine MAP3K3.", doc_rank=2),
]  # fmt: skip
Q = "What does MAP3K3 do in NF-kappaB signalling?"
V = OverlapVerifier()
X = RegexBiomedEntityExtractor()
CFG = VerificationConfig.from_yaml("configs/verification.yaml")


def run(gen: TemplateGenerator, evidence=EVIDENCE):
    return verify(answer(Q, evidence, gen), evidence, V, X, config=CFG)


def test_grounded_answer_is_fully_supported():
    r = run(TemplateGenerator())
    assert r.n_claims >= 1 and r.entailment_rate == 1.0 and r.contradiction_rate == 0.0
    assert r.citation_resolution_rate == 1.0 and r.citation_support_rate == 1.0
    assert r.fabricated_citations == () and r.conflicts == []
    assert r.numeric_fidelity in (None, 1.0) and r.entity_fidelity in (None, 1.0)
    assert r.verifier == V.name and r.policy_trace and "supported" in r.policy_trace[0]
    v0 = r.verdicts[0]
    assert v0.critical and v0.best_passage_id == "pmid:900001#0" and v0.supporting_ids


def test_fabricated_citation_fails_citation_check():
    r = run(TemplateGenerator(fabricate_citation="PMID:99999999"))
    assert "PMID:99999999" in r.fabricated_citations
    v0 = r.verdicts[0]
    assert v0.fabricated_ids == ("PMID:99999999",) and v0.supporting_ids == ()
    assert v0.citations_consistent is False and r.citation_support_rate < 1.0
    # the claim text is still true to the evidence, so support itself is high
    assert v0.label == "supported"


def test_entity_swap_fails_entity_check():
    r = run(TemplateGenerator(swap_entity=("MAP3K3", "MAP3K7")))
    bad = [v for v in r.verdicts if "map3k7" in v.entity_missing]
    assert bad, r.policy_trace
    assert r.entity_fidelity is not None and r.entity_fidelity < 1.0


def test_altered_number_fails_numeric_check():
    r = run(TemplateGenerator(alter_numbers=True))
    bad = [v for v in r.verdicts if v.numeric_mismatches]
    assert bad and any("71%" in m for v in bad for m in v.numeric_mismatches)
    assert r.numeric_fidelity is not None and r.numeric_fidelity < 1.0


def test_negation_flip_is_contradicted():
    r = run(TemplateGenerator(flip_negation=True))
    assert r.contradiction_rate > 0 and r.critical_contradicted >= 1
    v = next(v for v in r.verdicts if v.label == "contradicted")
    assert v.contradicting_passage_id is not None and v.contradiction >= CFG.contradict_threshold


def test_conflicting_sources_are_recorded_not_resolved():
    ev = EVIDENCE + [
        unit(900009, "MAP3K3 is not required for NF-kappaB signalling in hepatocytes.",
             year=2021, source_type=SourceType.RCT, doc_rank=3)
    ]  # fmt: skip
    r = run(TemplateGenerator(max_claims=1), evidence=ev)
    assert r.conflicts and r.conflicts[0].supporting_ids == ("PMID:900001",)
    assert "PMID:900009" in r.conflicts[0].contradicting_ids
    assert "2016–2021" in r.conflicts[0].description
    assert any("conflict on" in line for line in r.policy_trace)


def test_abstention_and_empty_answer_paths():
    r = run(TemplateGenerator(always_abstain=True))
    assert r.n_claims == 0 and r.policy_trace == ["generator abstained: insufficient_evidence"]
    r2 = verify(answer(Q, [], TemplateGenerator()), [], V, X)
    assert r2.n_claims == 0 and "no_evidence" in r2.policy_trace[0]


def test_calibrated_thresholds_override_yaml():
    cfg = CFG.with_thresholds({"entail": 0.95, "contradict": 0.99})
    assert cfg.entail_threshold == 0.95 and cfg.contradict_threshold == 0.99
    r = verify(answer(Q, EVIDENCE, TemplateGenerator()), EVIDENCE, V, X, config=CFG,
               thresholds={"entail": 0.999})  # fmt: skip
    assert r.entailment_rate == 0.0  # nothing clears an impossible bar; no silent pass
    assert CFG.with_thresholds(None) is CFG
    with pytest.raises(ValueError):
        VerificationConfig(entail_threshold=1.5)
