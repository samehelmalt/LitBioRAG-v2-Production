import json

from app.core.documents import SourceType
from app.evidence.selector import EvidenceUnit
from app.generation import prompts
from app.generation.contract import (
    ABSTAIN_FORMAT,
    ABSTAIN_INSUFFICIENT,
    ANSWER_SCHEMA,
    GeneratedAnswer,
    normalize_citation,
    parse_answer,
)

GOOD = {
    "answer": "MAP3K3 is required for NF-kappaB signalling.",
    "abstained": False,
    "claims": [
        {"text": "MAP3K3 is required for NF-kappaB signalling.", "citations": ["pmid: 900001"],
         "critical": True},
        {"text": "Knockdown reduced p65 translocation by 61%.", "citations": ["PMID:900001"],
         "value": 61, "unit": "%"},
    ],
    "conflicts": [],
}  # fmt: skip


def unit(**kw) -> EvidenceUnit:
    base = dict(passage_id="pmid:1#0", doc_id="pmid:1", pmid="1", doi=None, title="T", year=2016,
                journal="J", source_type=SourceType.JOURNAL_ARTICLE, is_peer_reviewed=True,
                text="Some evidence text.", char_start=0, char_end=19, score=1.0, doc_rank=1,
                url=None)  # fmt: skip
    base.update(kw)
    return EvidenceUnit(**base)


def test_normalize_citation():
    assert normalize_citation("pmid: 123") == "PMID:123"
    assert normalize_citation("DOI:10.1101/ABC") == "DOI:10.1101/abc"
    assert normalize_citation("Slide: L4-27") == "SLIDE:L4-27"
    assert normalize_citation("random") == "random"


def test_parse_good_and_roundtrip():
    a = parse_answer(json.dumps(GOOD), prompt_id="p", model="m")
    assert not a.abstained and a.format_ok and len(a.claims) == 2
    assert a.claims[0].id == "C1" and a.claims[0].citations == ["PMID:900001"]
    assert a.claims[0].critical and a.claims[1].value == 61 and a.claims[1].unit == "%"
    assert a.all_citations() == ["PMID:900001"]
    again = GeneratedAnswer.model_validate_json(a.model_dump_json())
    assert again == a


def test_parse_tolerates_fences_and_chatter_and_trailing_commas():
    raw = "Sure! Here is the JSON:\n```json\n" + json.dumps(GOOD)[:-1] + ",}\n```\nHope this helps."
    a = parse_answer(raw)
    assert a.format_ok and len(a.claims) == 2


def test_parse_failure_is_forced_abstention():
    a = parse_answer("I think MAP3K3 does something but I am not sure.")
    assert a.abstained and a.abstain_reason == ABSTAIN_FORMAT and not a.format_ok
    assert a.claims == [] and "could not produce" in a.answer
    b = parse_answer('["not", "an", "object"]')
    assert b.abstained and not b.format_ok


def test_no_claims_means_abstention_with_default_reason():
    a = parse_answer(json.dumps({"answer": "x", "abstained": False, "claims": []}))
    assert a.abstained and a.abstain_reason == ABSTAIN_INSUFFICIENT
    b = parse_answer(json.dumps({"answer": "", "abstained": True, "abstain_reason": "weird"}))
    assert b.abstain_reason == ABSTAIN_INSUFFICIENT and b.answer


def test_schema_has_required_fields():
    assert set(ANSWER_SCHEMA["required"]) == {"answer", "claims", "abstained"}
    assert "citations" in ANSWER_SCHEMA["properties"]["claims"]["items"]["required"]


def test_prompt_invariants_and_evidence_rendering():
    text = (prompts.SYSTEM + prompts.USER_TEMPLATE).lower()
    assert "must not abstain" not in text and "must answer" not in text
    assert "do not take a majority" in text  # conflicts are reported, never voted away
    assert "abstain" in text
    msgs = prompts.build_messages(
        "Does X do Y?",
        [unit(), unit(passage_id="doi:10.1/x#0", doc_id="doi:10.1/x", pmid=None, doi="10.1/x",
              source_type=SourceType.PREPRINT, is_peer_reviewed=False, year=None)],  # fmt: skip
    )
    assert msgs[0]["role"] == "system" and "abstain" in msgs[0]["content"].lower()
    user = msgs[1]["content"]
    assert "[PMID:1 | 2016 | journal_article | peer-reviewed]" in user
    assert "[DOI:10.1/x | n.d. | preprint | PREPRINT, not peer reviewed]" in user
    assert '"required"' in user and "Does X do Y?" in user
    assert prompts.render_evidence([]) == "(none)"
