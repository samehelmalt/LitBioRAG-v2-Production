"""Redis-backed state behaves exactly like the per-process stores, and two API apps sharing one
Redis enforce one limit (the multi-replica shape)."""

import time

import pytest

fakeredis = pytest.importorskip("fakeredis")

from app.api.trials import TrialLimitExceeded  # noqa: E402
from app.core.budget import BudgetExceeded, Pricing, Usage  # noqa: E402
from app.core.shared_state import (  # noqa: E402
    RedisJobTable,
    RedisRateLimiter,
    RedisTrialLedger,
    RedisUsageLedger,
    StateBackend,
)


@pytest.fixture
def r():
    return fakeredis.FakeRedis()


def test_rate_limiter_token_bucket(r):
    lim = RedisRateLimiter(r, rate_per_minute=60, burst=2)
    t = 1000.0
    assert lim.allow("u", t) == (True, 0.0) and lim.allow("u", t)[0]
    ok, retry = lim.allow("u", t)
    assert not ok and 0 < retry <= 1.0  # one token per second
    assert lim.allow("u", t + 1.0)[0]  # refilled
    assert lim.allow("other", t)[0]  # independent buckets


def test_trial_ledger_semantics_match_sqlite(r):
    led = RedisTrialLedger(r, limit=2)
    assert led.reserve("s1", "r1") == 1
    assert led.reserve("s1", "r1") == 1  # idempotent per request id
    assert led.reserve("s1", "r2") == 0
    with pytest.raises(TrialLimitExceeded):
        led.reserve("s1", "r3")
    with pytest.raises(TrialLimitExceeded):
        led.reserve("anonymous", "r9")
    with pytest.raises(TrialLimitExceeded):
        led.reserve("s2", "r1")  # a request id never changes hands
    assert led.refund("s1", "r2") and led.remaining("s1") == 1
    assert not led.refund("s2", "r1") and led.used("s1") == 1
    assert RedisTrialLedger(r, limit=2).remaining("s1") == 1  # shared, not per instance


def test_usage_ledger_daily_cap_is_shared(r, tmp_path):
    pricing = Pricing({"models": {"m": {"input": 1_000_000, "output": 0}}})  # 1 USD per token
    a = RedisUsageLedger(r, tmp_path / "a.jsonl", max_usd_per_day=2.5, pricing=pricing)
    b = RedisUsageLedger(r, tmp_path / "b.jsonl", max_usd_per_day=2.5, pricing=pricing)
    a.record(Usage("m", prompt_tokens=1), request_id="x")
    b.record(Usage("m", prompt_tokens=1), request_id="y")
    assert a.spent_today == pytest.approx(2.0) and b.spent_today == pytest.approx(2.0)
    with pytest.raises(BudgetExceeded):
        b.check_affordable(1.0)  # replica b sees replica a's spend
    assert (tmp_path / "a.jsonl").read_text().count("\n") == 1  # each replica keeps its audit


def test_job_table_interface(r):
    jobs = RedisJobTable(r)
    j = jobs.start("kb1:f1", kb_id="kb1", file_id="f1")
    assert j["status"] == "queued" and j["n_units"] is None and j["error"] is None
    jobs.update("kb1:f1", status="done", n_units=12, needs_ocr_units=0, finished_at=time.time(),
                ignored="x")  # fmt: skip
    j = jobs["kb1:f1"]
    assert j["status"] == "done" and j["n_units"] == 12 and isinstance(j["finished_at"], float)
    jobs.start("kb1:f2", kb_id="kb1", file_id="f2")
    assert [k for k, _ in jobs.items()] == ["kb1:f1", "kb1:f2"] and len(jobs.values()) == 2
    assert jobs.get("missing") is None
    with pytest.raises(KeyError):
        jobs["missing"]


def test_backend_selection(r):
    sq = StateBackend("sqlite")
    assert not sq.shared and sq.kb_jobs() is None
    assert type(sq.rate_limiter(10, 2)).__name__ == "RateLimiter"
    rd = StateBackend("redis", r)
    assert rd.shared and isinstance(rd.kb_jobs(), RedisJobTable)
    assert isinstance(rd.trial_ledger(None, 3), RedisTrialLedger)
    with pytest.raises(ValueError):
        StateBackend("redis")
    with pytest.raises(RuntimeError):
        StateBackend.from_settings("redis", "")


def test_two_api_replicas_share_limits(pipeline, tmp_path, monkeypatch, r):
    from fastapi.testclient import TestClient

    from app.api.main import create_app
    from app.api.settings import (
        ApiSettings,
        AuditSettings,
        AuthSettings,
        LimitSettings,
        TrialSettings,
    )
    from tests.api.conftest import SECRET, token

    monkeypatch.setenv("LITBIORAG_JWT_SECRET", SECRET)
    backend = StateBackend("redis", r)

    def app(name):
        st = ApiSettings(bundle=str(pipeline.bundle), profile="test", state_backend="redis",
                         trial=TrialSettings(questions=1),
                         auth=AuthSettings(required=False),
                         limits=LimitSettings(rate_per_minute=600, burst=1000),
                         audit=AuditSettings(path=str(tmp_path / f"{name}.jsonl")))  # fmt: skip
        return TestClient(create_app(st, pipeline=pipeline, state=backend))

    user = {"Authorization": f"Bearer {token(sub='prospect-7')}"}
    with app("a") as a, app("b") as b:
        assert a.get("/health").json()["state_backend"] == "redis"
        assert a.post("/api/v1/query", json={"question": "What does MAP3K3 do?"},
                      headers=user).status_code == 200
        r2 = b.post("/api/v1/query", json={"question": "What does MAP3K3 do?"}, headers=user)
        assert r2.status_code == 429 and "used up" in r2.json()["error"]  # replica b sees it
        assert b.get("/api/v1/trial", headers=user).json()["remaining"] == 0
