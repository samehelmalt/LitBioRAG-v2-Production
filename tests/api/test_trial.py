"""Trial allowance: reserved before the run, refunded on failure, persistent, admin-exempt."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.settings import ApiSettings, AuditSettings, AuthSettings, LimitSettings, TrialSettings
from app.api.trials import TrialLedger, TrialLimitExceeded
from app.core.budget import UsageLedger
from tests.api.conftest import SECRET, token

Q = "What does MAP3K3 do in NF-kappaB signalling?"


def test_ledger_semantics(tmp_path):
    led = TrialLedger(tmp_path / "t.sqlite", limit=2)
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
    again = TrialLedger(tmp_path / "t.sqlite", limit=2)  # survives restart
    assert again.remaining("s1") == 1 and again.remaining("s2") == 2


@pytest.fixture
def trial_client(pipeline, tmp_path, monkeypatch):
    monkeypatch.setenv("LITBIORAG_JWT_SECRET", SECRET)
    st = ApiSettings(bundle=str(pipeline.bundle), profile="test",
                     trial=TrialSettings(questions=2, db_path=str(tmp_path / "trial.sqlite")),
                     auth=AuthSettings(required=False),
                     limits=LimitSettings(rate_per_minute=600, burst=1000),
                     audit=AuditSettings(path=str(tmp_path / "audit.jsonl")))  # fmt: skip
    with TestClient(create_app(st, pipeline=pipeline, ledger=UsageLedger(None))) as c:
        yield c


def test_trial_enforced_at_the_api(trial_client):
    c = trial_client
    user = {"Authorization": f"Bearer {token(sub='prospect-1')}"}
    assert c.post("/api/v1/query", json={"question": Q}).status_code == 401  # anonymous
    assert c.get("/api/v1/trial", headers=user).json() == {
        "enabled": True, "limit": 2, "remaining": 2, "used": 0, "exempt": False}
    # a rejected request (unentitled course) costs nothing: validation runs before reservation
    assert c.post("/api/v1/query", json={"question": Q, "course_ids": ["IM-2026"]},
                  headers=user).status_code == 403
    assert c.get("/api/v1/trial", headers=user).json()["remaining"] == 2
    assert c.post("/api/v1/query", json={"question": Q}, headers=user).status_code == 200
    with c.stream("GET", "/api/v1/query/stream", params={"question": Q}, headers=user) as r:
        body = "".join(r.iter_text())
    assert "event: result" in body
    assert c.get("/api/v1/trial", headers=user).json()["remaining"] == 0
    r = c.post("/api/v1/query", json={"question": Q}, headers=user)
    assert r.status_code == 429 and "used up" in r.json()["error"]
    admin = {"Authorization": f"Bearer {token(sub='a', roles=['admin'])}"}
    assert c.post("/api/v1/query", json={"question": Q}, headers=admin).status_code == 200
    assert c.get("/api/v1/trial", headers=admin).json()["exempt"] is True


def test_trial_disabled_by_default(client):
    assert client.get("/api/v1/trial").json()["enabled"] is False
