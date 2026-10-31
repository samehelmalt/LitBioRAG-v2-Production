"""Token budget, usage ledger and spend caps."""

import io
import json
import urllib.error

import pytest

from app.core.budget import (
    BudgetExceeded,
    GeneratorBudget,
    HeuristicCounter,
    Pricing,
    ResourceConfig,
    Usage,
    UsageLedger,
    fit_evidence,
)
from app.core.documents import SourceType
from app.evidence.selector import EvidenceUnit
from app.generation.base import OpenAICompatibleGenerator, TemplateGenerator
from app.generation.contract import ABSTAIN_RESOURCE
from app.generation.service import GenerationConfig, answer


def unit(i: int, text: str) -> EvidenceUnit:
    return EvidenceUnit(passage_id=f"pmid:{i}#0", doc_id=f"pmid:{i}", pmid=str(i), doi=None,
                        title=f"T{i}", year=2020, journal="J", source_type=SourceType.RCT,
                        is_peer_reviewed=True, text=text, char_start=100, char_end=100 + len(text),
                        score=1.0, doc_rank=i, url=None)  # fmt: skip


SENT = "Atorvastatin 40 mg reduced LDL cholesterol by 39% at 12 weeks in 212 adults. "


def test_fit_keeps_rank_order_truncates_long_units_and_keeps_offsets():
    units = [unit(1, SENT * 2), unit(2, SENT * 40), unit(3, SENT * 3), unit(4, SENT * 3)]
    budget = GeneratorBudget(context_window=2048, reserve_output=256, prompt_overhead=200,
                             max_evidence_tokens=260, max_tokens_per_unit=120)  # fmt: skip
    kept, rep = fit_evidence(units, budget, HeuristicCounter())
    ids = [u.passage_id for u in kept]
    assert ids[0] == "pmid:1#0" and "pmid:2#0" in ids  # long unit truncated, not dropped
    assert "pmid:2#0" in rep.truncated
    long = next(u for u in kept if u.passage_id == "pmid:2#0")
    assert long.text.endswith("adults.") and long.char_end == 100 + len(long.text)
    assert rep.used_tokens <= rep.budget_tokens
    assert rep.dropped, "the budget was small enough to drop the tail"
    assert set(rep.kept) | set(rep.dropped) == {u.passage_id for u in units}


def test_budget_room_is_bounded_by_context_window():
    b = GeneratorBudget(context_window=1024, reserve_output=512, prompt_overhead=400,
                        max_evidence_tokens=3500)  # fmt: skip
    assert b.evidence_tokens == 112


def test_pricing_prefix_match_and_cost():
    p = Pricing({"as_of": "2026-10-31", "models": {"gpt-4o-mini": {"input": 0.15, "output": 0.6},
                                                   "gpt-4o": {"input": 2.5, "output": 10}}})
    u = p.price(Usage("openai_compatible:gpt-4o-mini-2024", 1_000_000, 1_000_000))
    assert u.cost_usd == pytest.approx(0.75) and u.pricing_as_of == "2026-10-31"
    assert p.price(Usage("hf_causal:Qwen/Qwen3-8B", 10**6, 10**6)).cost_usd == 0.0


def test_ledger_caps_and_replay(tmp_path):
    pricing = Pricing({"models": {"m": {"input": 1.0, "output": 1.0}}})
    led = UsageLedger(tmp_path / "ledger.jsonl", max_usd_per_request=0.5, max_usd_per_day=0.8,
                      pricing=pricing)  # fmt: skip
    led.record(Usage("x:m", 400_000, 0), request_id="r1")  # 0.40 USD
    assert led.spent_today == pytest.approx(0.4)
    with pytest.raises(BudgetExceeded) as e:
        led.check_affordable(0.45)  # day cap: 0.4 + 0.45 > 0.8 (below the request cap)
    assert e.value.scope == "day"
    with pytest.raises(BudgetExceeded) as e:
        led.check_affordable(0.6)  # request cap
    assert e.value.scope == "request"
    led.check_affordable(0.3)
    again = UsageLedger(tmp_path / "ledger.jsonl", pricing=pricing)  # replays today's spend
    assert again.spent_today == pytest.approx(0.4)
    rec = json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[0])
    assert rec["request_id"] == "r1" and rec["cost_usd"] == pytest.approx(0.4)


def test_service_abstains_with_resource_reason_when_cap_hit(tmp_path):
    pricing = Pricing({"models": {"evidence-extractive": {"input": 1000.0, "output": 1000.0}}})
    led = UsageLedger(None, max_usd_per_request=0.0001, pricing=pricing)
    out = answer("Does atorvastatin lower LDL?", [unit(1, SENT)], TemplateGenerator(), ledger=led)
    assert out.abstained and out.abstain_reason == ABSTAIN_RESOURCE
    assert "request" in out.usage["blocked_by"] and out.claims == []


def test_service_records_usage_and_fit():
    led = UsageLedger(None)
    out = answer("Does atorvastatin lower LDL?", [unit(1, SENT), unit(2, SENT * 30)],
                 TemplateGenerator(), config=GenerationConfig(),
                 budget=GeneratorBudget(max_tokens_per_unit=60), ledger=led)  # fmt: skip
    assert not out.abstained
    assert out.usage["prompt_tokens"] > 0 and out.usage["exact"] is False
    assert out.evidence_fit["truncated"] == ["pmid:2#0"]


def test_openai_adapter_reads_usage_and_retries(monkeypatch):
    calls = {"n": 0}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "rate limited", {}, None)
        body = {"choices": [{"message": {"content": '{"answer": "x", "claims": []}'}}],
                "usage": {"prompt_tokens": 321, "completion_tokens": 12}}  # fmt: skip
        return Resp(json.dumps(body).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("FAKE_KEY", "secret")
    g = OpenAICompatibleGenerator("m", endpoint="http://x/v1", api_key_env="FAKE_KEY",
                                  max_retries=2, backoff_s=0.0)  # fmt: skip
    text = g.generate([{"role": "user", "content": "hi"}])
    assert "answer" in text and calls["n"] == 2
    assert g.last_usage.prompt_tokens == 321 and g.last_usage.exact


def test_openai_adapter_gives_up_after_retries(monkeypatch):
    def always_503(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 503, "down", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", always_503)
    monkeypatch.setattr("time.sleep", lambda s: None)
    g = OpenAICompatibleGenerator("m", endpoint="http://x/v1", max_retries=1)
    with pytest.raises(urllib.error.HTTPError):
        g.generate([{"role": "user", "content": "hi"}])


def test_resource_config_loads_repo_defaults():
    cfg = ResourceConfig.from_yaml()
    b = cfg.generator_budget()
    assert b.evidence_tokens <= b.context_window
    assert cfg.gpu["residency"]["generator"] == "on_demand"
    assert cfg.cache["enabled"] is False
