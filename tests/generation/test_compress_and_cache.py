"""Token optimization: evidence compression keeps offsets valid; generation cache reuses answers;
schema stays compact and valid."""

import json

from app.core.cache import DiskCache
from app.core.documents import SourceType
from app.evidence.compress import compress_unit
from app.evidence.selector import EvidenceUnit
from app.generation.base import TemplateGenerator
from app.generation.contract import ANSWER_SCHEMA, schema_text
from app.generation.service import GenerationConfig, answer


def unit(text, start=1000):
    return EvidenceUnit(passage_id="pmid:1#0", doc_id="pmid:1", pmid="1", doi=None, title="T",
                        year=2020, journal="J", source_type=SourceType.RCT, is_peer_reviewed=True,
                        text=text, char_start=start, char_end=start + len(text), score=1.0,
                        doc_rank=1, url=None)  # fmt: skip


LONG = ("Background sentence one about trial design. Sentence two describes recruitment. "
        "Atorvastatin 40 mg reduced LDL cholesterol by 39% at 12 weeks. "
        "Adverse events were similar between arms. Myalgia occurred in 4% of participants. "
        "Sentence six about funding. Sentence seven about limitations. Sentence eight closes.")


def test_compression_keeps_relevant_window_and_offsets():
    u = unit(LONG)
    c = compress_unit(u, "Does atorvastatin lower LDL cholesterol?", max_sentences=3)
    assert "Atorvastatin 40 mg reduced LDL" in c.text and len(c.text) < len(u.text)
    assert LONG[c.char_start - 1000 : c.char_end - 1000] == c.text  # offsets point into source
    assert c.passage_id == u.passage_id and c.citation_id == u.citation_id


def test_compression_keeps_short_or_unrelated_passages_sensibly():
    short = unit("One sentence only.")
    assert compress_unit(short, "anything", max_sentences=4).text == short.text
    c = compress_unit(unit(LONG), "zebrafish fin regeneration", max_sentences=2)
    assert c.text.startswith("Background sentence one") and c.text.count(". ") <= 2


def test_service_reports_token_saving_and_caches_generations(tmp_path):
    cache = DiskCache(tmp_path / "c.sqlite")
    g = TemplateGenerator()
    cfg = GenerationConfig(max_sentences_per_unit=2)
    q = "Does atorvastatin lower LDL cholesterol?"
    a1 = answer(q, [unit(LONG)], g, config=cfg, cache=cache)
    assert a1.evidence_fit["evidence_tokens_sent"] < a1.evidence_fit["evidence_tokens_full"]
    assert a1.usage["cached"] is False
    a2 = answer(q, [unit(LONG)], g, config=cfg, cache=cache)
    assert a2.usage["cached"] is True and a2.usage["cost_usd"] == 0.0 and a2.answer == a1.answer
    # a different evidence span or model never hits the same entry
    a3 = answer(q, [unit(LONG, start=5)], g, config=cfg, cache=cache)
    assert a3.usage["cached"] is False
    a4 = answer(q, [unit(LONG)], g, config=GenerationConfig(cache_generations=False), cache=cache)
    assert a4.usage["cached"] is False


def test_schema_text_is_compact_and_equivalent():
    assert json.loads(schema_text()) == ANSWER_SCHEMA
    assert "\n" not in schema_text()
    assert len(schema_text()) < len(json.dumps(ANSWER_SCHEMA, indent=1))
