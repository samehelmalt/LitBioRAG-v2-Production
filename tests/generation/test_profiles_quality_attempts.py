"""Model-aware profiles, output-integrity checks and the bounded attempt ledger."""

from app.core.documents import SourceType
from app.evidence.selector import EvidenceUnit
from app.generation.attempts import ABSTAIN_GENERATION_QUALITY, run_attempts
from app.generation.base import TemplateGenerator
from app.generation.contract import parse_answer
from app.generation.profiles import select_profile
from app.generation.quality import assess, looks_truncated, repetition_ratio
from app.generation.service import GenerationConfig, answer
from app.verification.base import OverlapVerifier
from app.verification.entities import RegexBiomedEntityExtractor
from app.verification.pipeline import verify


def unit(pid, text):
    return EvidenceUnit(passage_id=f"pmid:{pid}#0", doc_id=f"pmid:{pid}", pmid=str(pid), doi=None,
                        title="T", year=2016, journal="J", source_type=SourceType.JOURNAL_ARTICLE,
                        is_peer_reviewed=True, text=text, char_start=0, char_end=len(text),
                        score=1.0, doc_rank=1, url=None)  # fmt: skip


EV = [unit(900001, "MAP3K3 (MEKK3) is required for NF-kappaB signalling in endothelial cells. "
                   "Knockdown of MAP3K3 reduced p65 nuclear translocation by 61%.")]
Q = "What does MAP3K3 do in NF-kappaB signalling?"
X, V = RegexBiomedEntityExtractor(), OverlapVerifier()


# ---- profiles ---------------------------------------------------------------------------------
def test_profiles_match_families_and_strip_reasoning():
    p = select_profile("Qwen/Qwen3-8B")
    assert p.name == "qwen3" and p.chat_template_kwargs == {"enable_thinking": False}
    assert p.decoding["no_repeat_ngram_size"] == 6
    d = select_profile("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    assert d.name == "deepseek-r1-distill" and d.max_new_tokens(500) == 1012
    final, reasoning = d.strip_reasoning(
        '<think>let me think</think>{"answer": "x"}<|end_of_sentence|>junk')
    assert final == '{"answer": "x"}' and reasoning == "let me think"
    final, reasoning = d.strip_reasoning('<think>never closed... ')
    assert final == "" and "never closed" in reasoning
    assert select_profile("some/unknown-model").name == "generic"
    custom = select_profile("my-model", [{"name": "mine", "match": "my-", "json_mode": False}])
    assert custom.name == "mine" and custom.json_mode is False


# ---- quality ----------------------------------------------------------------------------------
def test_quality_detects_repetition_role_echo_truncation_and_duplicates():
    rep_text = "MAP3K3 is required for signalling. " * 5
    assert repetition_ratio(rep_text) >= 0.6
    assert looks_truncated('{"answer": "x", "claims": [')
    claim = '{"text": "MAP3K3 is required for NF-kappaB signalling.", "citations": ["PMID:900001"]}'
    raw = ('{"answer": "Assistant: MAP3K3 is required for NF-kappaB signalling.", "claims": ['
           + claim + "," + claim + "]}")
    q = assess(raw, parse_answer(raw))
    assert not q.ok and q.role_echo and q.duplicate_claims == 1
    clean = '{"answer": "MAP3K3 is required for NF-kappaB signalling.", "claims": [' + claim + "]}"
    assert assess(clean, parse_answer(clean)).ok
    off = assess(clean, parse_answer(clean), question_entities={"brca1"},
                 claim_entities={"map3k3"})  # fmt: skip
    assert off.off_topic and not off.ok
    abst = ('{"answer": "", "abstained": true, "abstain_reason": "insufficient_evidence", '
            '"claims": []}')
    assert assess(abst, parse_answer(abst)).ok  # a deliberate abstention is not a defect


# ---- attempts ---------------------------------------------------------------------------------
def _runner(gen, max_attempts=2):
    def do_generate(feedback):
        parsed = answer(Q, EV, gen, config=GenerationConfig(max_attempts=max_attempts),
                        feedback=feedback)  # fmt: skip
        return parsed.raw, parsed

    def do_verify(parsed):
        return verify(parsed, EV, V, X, question=Q)

    return run_attempts(do_generate, do_verify, max_attempts=max_attempts,
                        question_entities=X.extract(Q), extract_entities=X.extract, units=EV)


def test_defective_first_attempt_is_recorded_and_second_attempt_selected():
    attempts, selected = _runner(TemplateGenerator(repeat_sentences=True))
    assert len(attempts) == 2
    assert not attempts[0].quality.ok and "duplicate" in " ".join(attempts[0].quality.defects)
    assert attempts[1].feedback_given.startswith("Your previous output was rejected")
    assert attempts[1].quality.ok and selected is attempts[1] and selected.selected
    assert attempts[0].raw != attempts[1].raw  # attempt 1 kept verbatim, not edited


def test_persistently_defective_output_abstains_with_generation_quality():
    attempts, selected = _runner(TemplateGenerator(role_echo=True, fix_on_feedback=False))
    assert selected.answer.abstained
    assert selected.answer.abstain_reason == ABSTAIN_GENERATION_QUALITY
    assert len([a for a in attempts if not a.selected]) == 2  # both real attempts kept for audit
    assert "role echo" in selected.veto_reason


def test_vetoed_attempt_triggers_feedback_with_the_fabricated_citation():
    gen = TemplateGenerator(fabricate_citation="PMID:999999", fix_on_feedback=False)
    attempts, selected = _runner(gen)
    assert attempts[0].vetoed and "fabricated" in attempts[0].veto_reason
    assert "PMID:999999" in attempts[1].feedback_given
    assert selected.answer.abstained  # still fabricated on attempt 2 → no release


def test_clean_first_attempt_stops_immediately_and_abstention_is_final():
    attempts, selected = _runner(TemplateGenerator())
    assert len(attempts) == 1 and selected.quality.ok and not selected.vetoed
    attempts, selected = _runner(TemplateGenerator(always_abstain=True))
    assert len(attempts) == 1 and selected.answer.abstained and selected.answer.format_ok


def test_pipeline_exposes_attempts(pipeline):
    out = pipeline.run(Q)
    assert out.attempts and out.attempts[-1].selected
    assert out.attempts[0].summary()["n"] == 1
