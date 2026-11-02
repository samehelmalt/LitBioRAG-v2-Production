"""The load-test script runs end to end against the in-process app (ASGI transport) and reports
latency, status mix and cache hits; the vllm-8b profile resolves and honours VLLM_URL; the
AEOS_CONCURRENCY_<ROLE> override reaches admission control."""

import json

import pytest

pytest.importorskip("httpx")

import httpx  # noqa: E402

from scripts import load_test  # noqa: E402
from tests.api.conftest import make_client  # noqa: E402


def test_load_test_smoke_on_the_fixture(pipeline, tmp_path, monkeypatch):
    monkeypatch.setenv("LITBIORAG_JWT_SECRET", "x" * 40)
    with make_client(pipeline, tmp_path) as tc:  # runs the lifespan (review/corpus services)
        app = tc.app
        health = tc.get("/health").json()

        def make_async_client():
            return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test", timeout=60)  # fmt: skip

        out = tmp_path / "load"
        summary = load_test.run(make_async_client, base_url="http://test", users=[2, 4],
                                per_user=3, pool=load_test.load_questions(None),
                                repeat_ratio=0.5, scope=None, out=out, health=health)  # fmt: skip
    assert [lv["users"] for lv in summary["levels"]] == [2, 4]
    top = summary["levels"][-1]
    assert top["requests"] == 12 and top["status"].get("200") == 12
    la = top["latency_ms"]
    assert la["p50"] is not None and la["p95"] >= la["p50"]
    assert top["answers_per_min"] > 0 and set(top["actions"]) <= {
        "ANSWER", "ANSWER_WITH_CAUTION", "CLARIFY", "ABSTAIN", "ESCALATE"}
    assert summary["profile"] == "test" and summary["state_backend"] == "sqlite"
    assert json.loads((out / "summary.json").read_text())["levels"][0]["users"] == 2
    report = (out / "report.md").read_text()
    assert "| users | answers/min |" in report and "| 4 |" in report


def test_pick_questions_respects_repeat_ratio():
    import random

    pool = [f"q{i}" for i in range(50)]
    asked = load_test.pick_questions(pool, 200, repeat_ratio=1.0, rng=random.Random(1))
    assert len(set(asked)) == 1  # after the first, every ask repeats
    asked = load_test.pick_questions(pool, 200, repeat_ratio=0.0, rng=random.Random(1))
    assert len(set(asked)) > 30
    assert load_test.percentile([], 50) is None and load_test.percentile([3.0, 1.0, 2.0], 50) == 2.0


def test_vllm_profile_and_concurrency_override(monkeypatch):
    from app.core.admission import Admission
    from models.registry.loader import ModelRegistry

    reg = ModelRegistry.load(profile="vllm-8b")
    spec = reg.spec("generator")
    assert spec.kind == "openai_compatible" and spec.endpoint == "http://vllm:8000/v1"
    assert reg.spec("dense_encoder").params["query"] == "ncbi/MedCPT-Query-Encoder"
    assert reg.spec("verifier").family != spec.family
    monkeypatch.setenv("VLLM_URL", "http://gpu-host:8000/v1")
    gen = reg.build("generator", cache=False)
    assert gen.endpoint == "http://gpu-host:8000/v1" and gen.api_key == ""
    monkeypatch.setenv("AEOS_CONCURRENCY_GENERATOR", "32")
    adm = Admission.from_config({"roles": {"generator": {"capacity": 1}, "verifier": {}}})
    assert adm.limits("generator").capacity == 32 and adm.limits("verifier").capacity == 1
    monkeypatch.setenv("AEOS_CONCURRENCY_RERANKER", "3")
    assert Admission.from_config({}).limits("reranker").capacity == 3  # role added by the env
