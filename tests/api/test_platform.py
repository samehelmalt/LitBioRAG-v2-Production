"""Chats with follow-up context, streamed progress, per-user API keys, tools, datasets, profiles."""

import json

from tests.api.conftest import token

U = {"Authorization": f"Bearer {token(sub='user-7')}"}
ADMIN = {"Authorization": f"Bearer {token(sub='admin-1', roles=['admin'])}"}
Q = "What does MAP3K3 do in NF-kappaB signalling?"


def test_chats_carry_history_as_context(client):
    assert client.get("/api/v1/chats").json() == []  # anonymous: none
    assert client.post("/api/v1/chats", json={}).status_code == 401
    chat = client.post("/api/v1/chats", json={}, headers=U).json()
    r1 = client.post("/api/v1/query", json={"question": Q, "chat_id": chat["chat_id"]}, headers=U)
    assert r1.status_code == 200 and r1.json()["chat_id"] == chat["chat_id"]
    r2 = client.post("/api/v1/query", json={"question": "And what about knockdown effects?",
                                            "chat_id": chat["chat_id"]}, headers=U)
    assert r2.status_code == 200
    full = client.get(f"/api/v1/chats/{chat['chat_id']}", headers=U).json()
    assert [m["question"] for m in full["messages"]] == [Q, "And what about knockdown effects?"]
    assert full["title"] == Q[:80] and full["messages"][1]["response"]["request_id"]
    lst = client.get("/api/v1/chats", headers=U).json()
    assert lst[0]["chat_id"] == chat["chat_id"]
    # another user cannot read or append
    other = {"Authorization": f"Bearer {token(sub='someone-else')}"}
    assert client.get(f"/api/v1/chats/{chat['chat_id']}", headers=other).status_code == 404
    assert client.post("/api/v1/query", json={"question": Q, "chat_id": chat["chat_id"]},
                       headers=other).status_code == 404
    assert client.delete(f"/api/v1/chats/{chat['chat_id']}", headers=U).status_code == 200


def test_history_is_rendered_into_the_prompt_as_context_only():
    from app.core.documents import SourceType
    from app.evidence.selector import EvidenceUnit
    from app.generation.prompts import build_messages

    u = EvidenceUnit(passage_id="pmid:1#0", doc_id="pmid:1", pmid="1", doi=None, title="T",
                     year=2020, journal=None, source_type=SourceType.RCT, is_peer_reviewed=True,
                     text="x", char_start=0, char_end=1, score=1.0, doc_rank=1,
                     url=None)  # fmt: skip
    plain = build_messages("q2", [u])[1]["content"]
    ctx = build_messages("q2", [u], ["first question about MAP3K3"])[1]["content"]
    assert "Conversation context" not in plain
    assert "Conversation context (NOT evidence)" in ctx and "- first question about MAP3K3" in ctx
    assert "NOT evidence" in ctx and ctx.endswith("Respond with the JSON object only.")


def test_stream_emits_stages_then_the_same_result(client):
    with client.stream("GET", "/api/v1/query/stream", params={"question": Q}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    events = [e for e in body.split("\n\n") if e.strip()]
    kinds = [e.split("\n")[0].replace("event: ", "") for e in events]
    assert kinds[-1] == "result" and kinds.count("stage") >= 4
    stages = [json.loads(e.split("\ndata: ")[1])["stage"] for e in events
              if e.startswith("event: stage")]  # fmt: skip
    assert stages[:2] == ["scope_check", "retrieving"] and "verifying" in stages
    result = json.loads(events[-1].split("\ndata: ")[1])
    assert result["action"] in ("ANSWER", "ANSWER_WITH_CAUTION") and result["sources"]


def test_stream_reports_errors_as_events(client):
    with client.stream("GET", "/api/v1/query/stream",
                       params={"question": Q, "course_ids": "IM-2026"}) as r:
        body = "".join(r.iter_text())
    assert "event: error" in body and '"status": 403' in body


def test_api_keys_act_as_their_owner(client):
    assert client.get("/api/v1/keys").status_code == 401
    k = client.post("/api/v1/keys", json={"name": "notebook"}, headers=U).json()
    assert k["key"].startswith("lbr_") and k["prefix"] == k["key"][:10]
    lst = client.get("/api/v1/keys", headers=U).json()
    assert lst[0]["prefix"] == k["prefix"] and lst[0]["revoked_at"] is None
    chat = client.post("/api/v1/chats", json={}, headers=U).json()  # owned by user-7
    r = client.post("/api/v1/query", json={"question": Q, "chat_id": chat["chat_id"]},
                    headers={"X-API-Key": k["key"]})
    assert r.status_code == 200  # the key reaches user-7's own chat
    assert client.post("/api/v1/keys", json={"name": "x"},
                       headers={"X-API-Key": k["key"]}).status_code == 401  # keys cannot mint keys
    assert client.delete(f"/api/v1/keys/{k['prefix']}", headers=U).status_code == 200
    assert client.get("/api/v1/models", headers={"X-API-Key": k["key"]}).status_code == 401


def test_research_tools(client):
    r = client.post("/api/v1/tools/entities", json={
        "text": "Simvastatin 20 mg caused myalgia; hyperkalemia was not observed.",
        "question": "Does atorvastatin cause hypokalemia?"}).json()
    pairs = {(c["text_entity"].lower(), c["asked_entity"].lower(), c["reason"])
             for c in r["confusable"]}  # fmt: skip
    assert ("simvastatin", "atorvastatin", "drug_class:statin") in pairs
    assert ("hyperkalemia", "hypokalemia", "antonym_prefix") in pairs
    v = client.post("/api/v1/tools/verify", json={
        "claim": "MAP3K3 is required for NF-kappaB signalling.",
        "passage": "We show that MAP3K3 (MEKK3) is required for NF-kappaB signalling."}).json()
    assert v["nli"]["entail"] > v["nli"]["contradict"] and v["entities_missing"] == []


def test_datasets_summary(client):
    d = client.get("/api/v1/datasets").json()
    assert d["splits"]["version"] == "v1" and d["splits"]["counts"]["test"] >= 24
    assert set(d["labels"]) == {"feedback", "review_labels"}


def test_admin_profiles_and_switch(client):
    assert client.get("/api/v1/admin/profiles").status_code == 403
    pr = client.get("/api/v1/admin/profiles", headers=ADMIN).json()
    assert pr["active"] == "test" and {"test", "hosted", "default"} <= set(pr["profiles"])
    assert pr["profiles"]["hosted"]["models"]["generator"].startswith("openai_compatible:")
    # switching to a profile whose models are not installed is refused, service unchanged
    r = client.post("/api/v1/admin/profile", json={"profile": "no-such-profile"}, headers=ADMIN)
    assert r.status_code == 409
    r = client.post("/api/v1/admin/profile", json={"profile": "test"}, headers=ADMIN)
    assert r.status_code == 200 and r.json()["active"] == "test"
    assert client.post("/api/v1/query", json={"question": Q}).status_code == 200
