"""Landing page: owner copy endpoint, trial-request leads (validation, honeypot, admin list), and
self-service demo tokens (disabled by default, rate limited, usable for queries, trial-limited)."""

import pytest
from fastapi.testclient import TestClient

from app.api.leads import LeadStore
from app.api.main import create_app
from app.api.settings import ApiSettings, AuditSettings, AuthSettings, LimitSettings, TrialSettings
from app.core.budget import UsageLedger
from tests.api.conftest import SECRET, token

LEAD = {"name": "Dr Example", "email": "dean@example.edu", "institution": "Example University",
        "role": "faculty", "path": "institution", "course_size": "300", "message": "Pharmacology"}


def test_landing_copy_is_public_and_demo_off_by_default(client):
    c = client.get("/api/v1/landing").json()
    assert c["organisation"] == "AEOS Medical" and len(c["shapes"]) == 4
    assert all(s["price"] == "on request" for s in c["shapes"])  # no invented prices
    assert c["demo_enabled"] is False and c["demo_questions"] is None
    assert client.post("/api/v1/demo/token").status_code == 404


def test_leads_validation_honeypot_and_admin_list(client, tmp_path):
    r = client.post("/api/v1/leads", json=LEAD)
    assert r.status_code == 202 and r.json()["lead_id"].startswith("lead_")
    assert client.post("/api/v1/leads", json={**LEAD, "email": "not-an-email"}).status_code == 422
    assert client.post("/api/v1/leads", json={**LEAD, "role": "ceo"}).status_code == 422
    bot = client.post("/api/v1/leads", json={**LEAD, "website": "http://spam"})
    assert bot.status_code == 202 and "lead_id" not in bot.json()
    assert client.get("/api/v1/leads").status_code == 403  # anonymous
    assert client.get("/api/v1/leads", headers={
        "Authorization": f"Bearer {token(sub='u')}"}).status_code == 403
    rows = client.get("/api/v1/leads", headers={
        "Authorization": f"Bearer {token(sub='a', roles=['admin'])}"}).json()
    assert len(rows) == 1 and rows[0]["email"] == LEAD["email"] and "website" not in rows[0]
    assert rows[0]["status"] == "new"
    audit = client.audit_path.read_text()
    assert '"event": "lead"' in audit and "dean@example.edu" not in audit  # audit has no PII


def test_lead_store_without_path_and_webhook_failure_is_swallowed(monkeypatch):
    from app.api.leads import LeadIn

    store = LeadStore(None)
    monkeypatch.setenv("LEADS_WEBHOOK_URL", "http://127.0.0.1:9/unreachable")
    rec = store.add(LeadIn(**LEAD), request_id="r1")
    assert rec["name"] == "Dr Example" and store.list() == []


@pytest.fixture
def demo_client(pipeline, tmp_path, monkeypatch):
    monkeypatch.setenv("LITBIORAG_JWT_SECRET", SECRET)
    st = ApiSettings(bundle=str(pipeline.bundle), profile="test", demo_tokens=True,
                     trial=TrialSettings(questions=2, db_path=str(tmp_path / "trial.sqlite")),
                     auth=AuthSettings(required=True),
                     limits=LimitSettings(rate_per_minute=600, burst=1000),
                     audit=AuditSettings(path=str(tmp_path / "audit.jsonl")))  # fmt: skip
    with TestClient(create_app(st, pipeline=pipeline, ledger=UsageLedger(None))) as c:
        yield c


def test_demo_token_flow(demo_client):
    c = demo_client
    copy = c.get("/api/v1/landing").json()
    assert copy["demo_enabled"] is True and copy["demo_questions"] == 2
    assert c.post("/api/v1/query", json={"question": "What does MAP3K3 do?"}).status_code == 401
    d = c.post("/api/v1/demo/token").json()
    assert d["sub"].startswith("demo-") and d["questions"] == 2
    h = {"Authorization": f"Bearer {d['token']}"}
    assert c.get("/api/v1/trial", headers=h).json()["remaining"] == 2
    for _ in range(2):
        assert c.post("/api/v1/query", json={"question": "What does MAP3K3 do?"},
                      headers=h).status_code == 200
    assert c.post("/api/v1/query", json={"question": "What does MAP3K3 do?"},
                  headers=h).status_code == 429  # trial allowance applies to demo visitors
    for _ in range(2):
        c.post("/api/v1/demo/token")
    r = c.post("/api/v1/demo/token")
    assert r.status_code == 429 and "Retry-After" in r.headers  # 3 tokens per address


def test_demo_token_needs_a_real_secret(pipeline, tmp_path, monkeypatch):
    monkeypatch.setenv("LITBIORAG_JWT_SECRET", "short")
    st = ApiSettings(bundle=str(pipeline.bundle), profile="test", demo_tokens=True,
                     auth=AuthSettings(required=False),
                     audit=AuditSettings(path=str(tmp_path / "audit.jsonl")))  # fmt: skip
    with TestClient(create_app(st, pipeline=pipeline, ledger=UsageLedger(None))) as c:
        assert c.post("/api/v1/demo/token").status_code == 503
