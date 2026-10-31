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


def test_query_grounding_without_question_entities_is_neutral():
    qg = query_entity_grounding("Is fasting before sedation necessary?", [("C1", "x", True)], [], X)
    assert qg.grounding is None and not qg.substitution_detected
