"""Entity substitution must be detected even when the substituted claim is perfectly cited."""

from app.core.documents import SourceType
from app.core.symbols import is_near_neighbour, normalize, symbol_like_tokens
from app.evidence.selector import EvidenceUnit
from app.generation.contract import Claim, GeneratedAnswer
from app.verification.base import OverlapVerifier
from app.verification.entities import RegexBiomedEntityExtractor, query_entity_grounding
from app.verification.pipeline import verify


def unit(pid, text, **kw) -> EvidenceUnit:
    base = dict(passage_id=f"pmid:{pid}#0", doc_id=f"pmid:{pid}", pmid=str(pid), doi=None,
                title=f"Title {pid}", year=2016, journal="J",
                source_type=SourceType.JOURNAL_ARTICLE, is_peer_reviewed=True, text=text,
                char_start=0, char_end=len(text), score=1.0, doc_rank=1, url=None)  # fmt: skip
    base.update(kw)
    return EvidenceUnit(**base)


EV = [
    unit(900001, "MAP3K3 (MEKK3) is required for NF-kappaB signalling in endothelial cells."),
    unit(900002, "MAP3K7, also known as TAK1, mediates TLR signalling.", doc_rank=2),
]
V, X = OverlapVerifier(), RegexBiomedEntityExtractor()


def test_symbol_helpers():
    assert symbol_like_tokens("What does MAP3K3 do in NF-kappaB signalling? TAS-102") == [
        "TAS-102", "MAP3K3"
    ]
    assert is_near_neighbour("MAP3K3", "MAP3K7") and is_near_neighbour("MAP3K3", "MAPK3")
    assert is_near_neighbour("atorvastatin", "simvastatin")
    assert not is_near_neighbour("MAP3K3", "MAP3K3") and not is_near_neighbour("MAP3K3", "TLR4")
    assert normalize("heart attack") == normalize("myocardial infarction") == normalize("MI")
    assert normalize("TAK1") == normalize("MAP3K7")  # alias, not a substitution


def test_perfectly_cited_wrong_gene_is_flagged():
    # the claim is TRUE and correctly cited — but it answers a different gene than asked
    ans = GeneratedAnswer(
        answer="MAP3K7 mediates TLR signalling.",
        claims=[Claim(id="C1", text="MAP3K7 mediates TLR signalling.", citations=["PMID:900002"],
                      critical=True)],  # fmt: skip
        question="What does MAP3K3 do in NF-kappaB signalling?",
    )
    r = verify(ans, EV, V, X)
    v = r.verdicts[0]
    assert v.label == "supported" and v.citations_consistent  # per-claim checks all pass...
    assert v.near_neighbour_substitution and v.substituted_for == "map3k7->map3k3"  # ...caught here
    assert r.substitution_detected and "map3k3" in r.question_entities_missing_in_claims
    assert any("ENTITY SUBSTITUTION" in line for line in r.policy_trace)


def test_correct_gene_is_not_flagged_and_alias_is_accepted():
    ans = GeneratedAnswer(
        answer="MAP3K3 is required for NF-kappaB signalling.",
        claims=[Claim(id="C1", text="MEKK3 is required for NF-kappaB signalling.",
                      citations=["PMID:900001"], critical=True)],  # fmt: skip
        question="What does MAP3K3 do in NF-kappaB signalling?",
    )
    r = verify(ans, EV, V, X)
    assert not r.substitution_detected
    assert r.question_entities_missing_in_claims == ()  # MEKK3 is an alias of MAP3K3
    assert r.query_entity_grounding == 1.0


def test_synonym_disease_terms_are_not_substitutions():
    ev = [unit(1, "Myocardial infarction was associated with elevated troponin levels.")]
    ans = GeneratedAnswer(
        answer="Heart attack is associated with elevated troponin.",
        claims=[Claim(id="C1", text="Heart attack is associated with elevated troponin levels.",
                      citations=["PMID:1"], critical=True)],  # fmt: skip
        question="Is myocardial infarction associated with troponin?",
    )
    r = verify(ans, ev, V, X)
    assert not r.substitution_detected


def test_drug_analogue_substitution():
    ev = [unit(7, "Atorvastatin reduced LDL cholesterol by 39% in the treatment arm."),
          unit(8, "Simvastatin reduced LDL cholesterol by 30% in the treatment arm.")]  # fmt: skip
    ans = GeneratedAnswer(
        answer="Simvastatin reduced LDL by 30%.",
        claims=[Claim(id="C1", text="Simvastatin reduced LDL cholesterol by 30%.",
                      citations=["PMID:8"], critical=True)],  # fmt: skip
        question="How much does atorvastatin reduce LDL cholesterol?",
    )
    r = verify(ans, ev, V, X)
    assert r.substitution_detected and r.verdicts[0].substituted_for == "simvastatin->atorvastatin"


def test_neighbour_claim_beside_a_correct_answer_is_off_target_not_substitution():
    ans = GeneratedAnswer(
        answer="...",
        claims=[
            Claim(id="C1", text="MAP3K3 is required for NF-kappaB signalling.",
                  citations=["PMID:900001"], critical=True),
            Claim(id="C2", text="MAP3K7, also known as TAK1, mediates TLR signalling.",
                  citations=["PMID:900002"], critical=False),
        ],  # fmt: skip
        question="What does MAP3K3 do in NF-kappaB signalling?",
    )
    r = verify(ans, EV, V, X)
    assert not r.substitution_detected
    by = {v.claim_id: v for v in r.verdicts}
    assert by["C2"].off_target and not by["C1"].off_target
    assert any("off-target" in line for line in r.policy_trace)


def test_query_grounding_without_question_entities_is_neutral():
    qg = query_entity_grounding("Is fasting before sedation necessary?", [("C1", "x", True)], [], X)
    assert qg.grounding is None and not qg.substitution_detected


# ------------------------------------------------------------------ beyond gene symbols ---------
import pytest  # noqa: E402

GENERAL_EV = {
    "drug": (
        unit(910001, "Atorvastatin 40 mg reduced LDL cholesterol by 39% at 12 weeks."),
        unit(910002, "Simvastatin 20 mg was associated with myalgia in 8% of participants.",
             doc_rank=2),  # fmt: skip
    ),
    "condition_prefix": (
        unit(910003, "Hypokalemia prolongs the QT interval and predisposes to arrhythmia."),
        unit(910004, "Hyperkalemia causes peaked T waves and can progress to asystole.",
             doc_rank=2),  # fmt: skip
    ),
    "numbered_condition": (
        unit(910005, "Type 2 diabetes is characterised by insulin resistance in adults."),
        unit(910006, "Type 1 diabetes results from autoimmune destruction of beta cells.",
             doc_rank=2),  # fmt: skip
    ),
    "letter_variant": (
        unit(910007, "Hepatitis B is transmitted perinatally and is preventable by vaccination."),
        unit(910008, "Hepatitis C is transmitted by blood and is curable with antivirals.",
             doc_rank=2),  # fmt: skip
    ),
    "lookalike_drug": (
        unit(910009, "Hydroxyzine is an antihistamine used for anxiety and pruritus."),
        unit(910010, "Hydralazine is a vasodilator used in hypertensive emergencies.",
             doc_rank=2),  # fmt: skip
    ),
}
GENERAL_CASES = [
    ("drug", "Does atorvastatin lower LDL cholesterol?",
     "Simvastatin 20 mg was associated with myalgia in 8% of participants.", "PMID:910002"),
    ("condition_prefix", "What ECG changes does hypokalemia cause?",
     "Hyperkalemia causes peaked T waves and can progress to asystole.", "PMID:910004"),
    ("numbered_condition", "What characterises type 2 diabetes?",
     "Type 1 diabetes results from autoimmune destruction of beta cells.", "PMID:910006"),
    ("letter_variant", "How is hepatitis B transmitted?",
     "Hepatitis C is transmitted by blood and is curable with antivirals.", "PMID:910008"),
    ("lookalike_drug", "What is hydroxyzine used for?",
     "Hydralazine is a vasodilator used in hypertensive emergencies.", "PMID:910010"),
]


@pytest.mark.parametrize("key,question,claim,cite", GENERAL_CASES)
def test_substitution_detected_for_every_entity_kind(key, question, claim, cite):
    ans = GeneratedAnswer(answer=claim, question=question,
                          claims=[Claim(id="C1", text=claim, citations=[cite],
                                        critical=True)])  # fmt: skip
    r = verify(ans, list(GENERAL_EV[key]), V, X, question=question)
    assert r.verdicts[0].label == "supported", "the attack claim is true and well cited"
    assert r.substitution_detected, (key, r.policy_trace)


@pytest.mark.parametrize("key,question,_claim,_cite", GENERAL_CASES)
def test_correct_entity_never_flagged(key, question, _claim, _cite):
    good = GENERAL_EV[key][0]
    ans = GeneratedAnswer(answer=good.text, question=question,
                          claims=[Claim(id="C1", text=good.text, citations=[good.citation_id],
                                        critical=True)])  # fmt: skip
    r = verify(ans, list(GENERAL_EV[key]), V, X, question=question)
    assert not r.substitution_detected, (key, r.policy_trace)
    asked = {m.norm for m in X.mentions(question) if m.kind != "gene"}
    assert not asked & set(r.question_entities_missing_in_claims)
