import json

from app.core.documents import SourceType
from app.evidence.selector import EvidenceUnit
from app.generation.base import OpenAICompatibleGenerator, TemplateGenerator
from app.generation.contract import ABSTAIN_FORMAT, ABSTAIN_NO_EVIDENCE
from app.generation.service import GenerationConfig, answer
from models.registry import ModelRegistry
from models.registry.loader import DEFAULT_PATH


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


def test_no_evidence_abstains_without_calling_model():
    class Boom:
        name = "boom"

        def generate(self, *a, **k):
            raise AssertionError("must not be called")

    a = answer(Q, [], Boom())
    assert a.abstained and a.abstain_reason == ABSTAIN_NO_EVIDENCE and a.model == "boom"


def test_template_generator_produces_grounded_claims():
    a = answer(Q, EVIDENCE, TemplateGenerator())
    assert not a.abstained and a.format_ok and a.claims
    assert a.claims[0].critical and "MAP3K3" in a.claims[0].text
    assert all(c.citations == ["PMID:900001"] or c.citations == ["PMID:900002"] for c in a.claims)
    assert a.unknown_citations == [] and a.prompt_id == "answer-json-v1"


def test_fabricated_citation_is_flagged_not_dropped():
    a = answer(Q, EVIDENCE, TemplateGenerator(fabricate_citation="PMID:99999999"))
    assert "PMID:99999999" in a.unknown_citations
    assert a.claims[0].citations == ["PMID:99999999"]  # kept so the verifier can fail it


def test_adversarial_knobs_change_text_deterministically():
    altered = answer(Q, EVIDENCE, TemplateGenerator(alter_numbers=True))
    assert any("71%" in c.text for c in altered.claims)
    swapped = answer(Q, EVIDENCE, TemplateGenerator(swap_entity=("MAP3K3", "MAP3K7")))
    assert all("MAP3K3" not in c.text for c in swapped.claims)
    flipped = answer(Q, EVIDENCE, TemplateGenerator(flip_negation=True))
    assert any("is not required" in c.text for c in flipped.claims)
    assert answer(Q, EVIDENCE, TemplateGenerator()) == answer(Q, EVIDENCE, TemplateGenerator())


def test_invalid_json_is_forced_abstention():
    a = answer(Q, EVIDENCE, TemplateGenerator(emit_invalid_json=True))
    assert a.abstained and a.abstain_reason == ABSTAIN_FORMAT and not a.format_ok


def test_registry_builds_generator_from_test_profile():
    reg = ModelRegistry.load(DEFAULT_PATH, profile="test", env={})
    gen = reg.build("generator")
    assert gen.name.startswith("template:")
    a = answer(Q, EVIDENCE, gen, config=GenerationConfig.from_yaml("configs/generation.yaml"))
    assert not a.abstained
    # swapping the generator kind is a spec change only
    reg2 = ModelRegistry.load(
        DEFAULT_PATH, profile="test",
        env={"LITBIORAG_MODEL__GENERATOR": "openai_compatible:my/model?endpoint=http://h:1/v1"},
    )  # fmt: skip
    g2 = reg2.build("generator")
    assert isinstance(g2, OpenAICompatibleGenerator) and g2.endpoint == "http://h:1/v1"


def test_openai_compatible_request_shape(monkeypatch):
    captured = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": '{"answer":"x"}'}}]}).encode()

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        captured["auth"] = req.get_header("Authorization")
        return Resp()

    monkeypatch.setenv("TEST_KEY", "sk-test")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    g = OpenAICompatibleGenerator("m", endpoint="http://srv/v1/", api_key_env="TEST_KEY")
    out = g.generate([{"role": "user", "content": "hi"}], max_tokens=5, temperature=0.0)
    assert out == '{"answer":"x"}' and captured["url"] == "http://srv/v1/chat/completions"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["auth"] == "Bearer sk-test" and captured["body"]["max_tokens"] == 5
