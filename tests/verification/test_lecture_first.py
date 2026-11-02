"""Lecture-first policy: course material is primary; literature differences are reported, never
silently override the lecture; outside the education profile the usual contradiction rules hold."""

from app.core.documents import SourceType
from app.evidence.selector import EvidenceUnit, _rebalance
from app.generation.contract import Claim, GeneratedAnswer
from app.generation.prompts import build_messages, render_evidence
from app.safety.gate import Action, decide
from app.verification.base import OverlapVerifier
from app.verification.entities import RegexBiomedEntityExtractor
from app.verification.pipeline import VerificationConfig, verify

V, X = OverlapVerifier(), RegexBiomedEntityExtractor()


def slide(n, text):
    return EvidenceUnit(passage_id=f"file:deck1:s{n}#0", doc_id=f"file:deck1:s{n}", pmid=None,
                        doi=None, title=f"Slide {n}", year=2026, journal="Potassium disorders",
                        source_type=SourceType.COURSE_SLIDE, is_peer_reviewed=False, text=text,
                        char_start=0, char_end=len(text), score=1.0, doc_rank=1, url=None,
                        kb_id="kb1", file_id="deck1", file_name="k.pptx", locator=f"s{n}",
                        locator_label=f"slide {n}", lecture_id="L03",
                        lecture_title="Potassium disorders", slide_number=n,
                        deck_version="2026a")  # fmt: skip


def paper(pid, text, rank=2):
    return EvidenceUnit(passage_id=f"pmid:{pid}#0", doc_id=f"pmid:{pid}", pmid=str(pid), doi=None,
                        title="Paper", year=2024, journal="J", source_type=SourceType.RCT,
                        is_peer_reviewed=True, text=text, char_start=0, char_end=len(text),
                        score=0.9, doc_rank=rank, url=None)  # fmt: skip


CLAIM = "Hyperkalemia is treated with calcium gluconate first."
EV = [slide(4, "Hyperkalemia is treated with calcium gluconate first."),
      paper(777, "Hyperkalemia is not treated with calcium gluconate first in this cohort.")]
Q = "How is hyperkalemia treated first?"


def _answer():
    return GeneratedAnswer(answer=CLAIM, question=Q,
                          claims=[Claim(id="C1", text=CLAIM, citations=["SLIDE:L03:4"],
                                        critical=True)])  # fmt: skip


def test_lecture_first_reports_the_difference_and_delivers_with_caution():
    rep = verify(_answer(), EV, V, X, config=VerificationConfig(lecture_first=True), question=Q)
    v = rep.verdicts[0]
    assert v.label == "supported" and rep.conflicts
    k = rep.conflicts[0]
    assert k.kind == "lecture_vs_literature"
    assert k.supporting_tiers == ("course_material",) and k.contradicting_tiers == ("literature",)
    d = decide(rep, _answer(), sufficiency=0.9, question=Q)
    assert d.action == Action.ANSWER_WITH_CAUTION and d.rule == "R11-conflict"
    assert "course material and the literature differ" in d.reason


def test_without_lecture_first_the_contradiction_vetoes():
    rep = verify(_answer(), EV, V, X, config=VerificationConfig(lecture_first=False), question=Q)
    assert rep.verdicts[0].label == "contradicted"
    d = decide(rep, _answer(), sufficiency=0.9, question=Q)
    assert d.action == Action.ABSTAIN and d.rule == "R5-critical-contradicted"


def test_literature_supported_claim_contradicted_by_lecture_is_still_contradicted():
    ev = [paper(777, CLAIM, rank=1), slide(4, "Hyperkalemia is not treated with calcium "
                                                "gluconate first.")]
    ans = GeneratedAnswer(answer=CLAIM, question=Q,
                          claims=[Claim(id="C1", text=CLAIM, citations=["PMID:777"],
                                        critical=True)])  # fmt: skip
    rep = verify(ans, ev, V, X, config=VerificationConfig(lecture_first=True), question=Q)
    assert rep.verdicts[0].label == "contradicted"  # the lecture is primary, not the paper


def test_prompt_groups_course_material_first_and_states_the_policy():
    msgs = build_messages(Q, EV, lecture_first=True)
    assert "Course-material policy" in msgs[0]["content"]
    body = msgs[1]["content"]
    assert body.index("=== Course material (primary source) ===") < body.index("=== Literature")
    assert "SLIDE:L03:4" in body and "PMID:777" in body
    plain = render_evidence(EV)
    assert "=== Course material" not in plain


def test_rebalance_reserves_the_course_share():
    units = [paper(1, "a", 1), paper(2, "b", 2), paper(3, "c", 3), paper(4, "d", 4),
             slide(1, "s1"), slide(2, "s2"), slide(3, "s3")]  # fmt: skip
    out = _rebalance(units, max_units=5, course_share=0.6)
    assert len(out) == 5 and sum(u.is_lecture_slide for u in out) == 3
    assert [u.passage_id for u in out] == [u.passage_id for u in units if u in out]  # rank order
    out = _rebalance(units[:4], max_units=3, course_share=0.6)  # no course material retrieved
    assert len(out) == 3 and not any(u.is_lecture_slide for u in out)
