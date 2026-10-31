import pytest

from app.core.verdicts import ClaimVerdict, VerificationReport
from app.generation.contract import Claim, GeneratedAnswer
from app.verification.base import OverlapVerifier
from app.verification.claims import is_critical, prepare_claims
from evaluation.metrics.claims import ClaimVerdict as MetricVerdict
from evaluation.metrics.claims import summarize
from models.registry import ModelRegistry
from models.registry.loader import DEFAULT_PATH

P = ("MAP3K3 (MEKK3) is required for NF-kappaB signalling. Knockdown of MAP3K3 reduced p65 "
     "nuclear translocation by 61%.")  # fmt: skip


def test_overlap_verifier_entail_neutral_contradict():
    v = OverlapVerifier()
    e, n, c = v.score_pairs([
        (P, "MAP3K3 is required for NF-kappaB signalling."),
        (P, "Aspirin prevents migraine in adolescents."),
        (P, "MAP3K3 is not required for NF-kappaB signalling."),
    ])  # fmt: skip
    assert e.entail > 0.8 and e.contradict < 0.1
    assert n.neutral > 0.6 and n.entail < 0.5
    assert c.contradict > 0.8 and c.entail < 0.1
    # negated premise + negated hypothesis agree
    neg = "This study does not examine MAP3K3."
    s = v.score_one(neg, "The study does not examine MAP3K3.")
    assert s.entail > 0.8


def test_verdict_models_are_shared_between_runtime_and_evaluation():
    assert MetricVerdict is ClaimVerdict
    v1 = ClaimVerdict(claim_id="C1", text="a", label="supported", cited_ids=("PMID:1",),
                      resolved_ids=("PMID:1",), supporting_ids=("PMID:1",), critical=True,
                      numeric_checked=1, numeric_ok=1)  # fmt: skip
    v2 = ClaimVerdict(claim_id="C2", text="b", label="contradicted", cited_ids=("PMID:9",),
                      fabricated_ids=("PMID:9",), critical=True)  # fmt: skip
    v3 = ClaimVerdict(claim_id="C3", text="c", label="unsupported")
    s = summarize([v1, v2, v3])
    r = VerificationReport.from_verdicts([v1, v2, v3], verifier="overlap")
    assert s.entailment_rate == r.entailment_rate == pytest.approx(1 / 3)
    assert s.critical_contradicted == r.critical_contradicted == 1
    assert r.citation_resolution_rate == 0.5 and r.citation_support_rate == 0.5
    assert r.fabricated_citations == ("PMID:9",) and r.uncited_claim_rate == pytest.approx(1 / 3)
    assert r.numeric_fidelity == 1.0 and r.entity_fidelity is None
    assert v1.citations_consistent is True and v2.citations_consistent is False
    assert v3.citations_consistent is None and v1.numeric_consistent is True
    with pytest.raises(ValueError):
        ClaimVerdict(label="maybe")
    assert VerificationReport.from_verdicts([]).n_claims == 0


def test_prepare_claims_splits_sentences_and_marks_critical():
    a = GeneratedAnswer(
        claims=[
            Claim(id="C1", text="MAP3K3 is required for NF-kappaB signalling. Knockdown reduced "
                  "translocation by 61%.", citations=["PMID:900001"]),
            Claim(id="C2", text="MAP3K7 may mediate TLR signalling.", citations=["PMID:900002"]),
            Claim(id="C3", text="Short.", citations=[]),
        ]
    )  # fmt: skip
    cs = prepare_claims(a)
    assert [c.id for c in cs] == ["C1.1", "C1.2", "C2"]
    assert all(c.citations == ["PMID:900001"] for c in cs[:2])
    assert cs[0].critical and cs[1].critical  # polarity word / number
    assert cs[2].modality == "hedged" and not is_critical("The authors describe a cohort.")
    assert not is_critical("MAP3K7, also known as TAK1, mediates TLR signalling.")  # symbol digits
    assert is_critical("Mortality fell by 20%.") and is_critical("MAP3K3 is not required.")
    assert prepare_claims(a, max_claims=1) == cs[:1]


def test_registry_builds_verifier_double_and_enforces_family_rule():
    reg = ModelRegistry.load(DEFAULT_PATH, profile="test", env={})
    ver = reg.build("verifier")
    assert ver.name.startswith("overlap")
    default = ModelRegistry.load(DEFAULT_PATH, profile="default", env={})
    assert default.spec("verifier").family == "deberta"
    assert default.spec("generator").family == "qwen"
