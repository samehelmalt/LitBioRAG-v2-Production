"""One request → one immutable record; every channel agrees; chats carry delivered turns only."""

import sqlite3

import pytest

from app.service.records import RecordStore, history_as_context
from tests.api.conftest import token

U = {"Authorization": f"Bearer {token(sub='user-9')}"}
REV = {"Authorization": f"Bearer {token(sub='dr-r', roles=['reviewer'])}"}
Q = "What does MAP3K3 do in NF-kappaB signalling?"
ESC = "My mother takes warfarin 5 mg; what dose should she take tonight for her INR?"


def test_record_channels_agree_with_the_response(client):
    r = client.post("/api/v1/query", json={"question": Q}, headers=U).json()
    rid = r["record_id"]
    assert rid == r["request_id"] and r["attempts"] and r["attempts"][-1]["selected"]
    mine = client.get(f"/api/v1/records/{rid}", headers=U).json()
    assert set(mine) >= {"user", "history", "control"} and "raw" not in mine
    assert mine["user"]["answer"] == r["answer"] and mine["user"]["action"] == r["action"]
    assert mine["control"]["versions"]["bundle_id"] == r["versions"]["bundle_id"]
    assert client.get(f"/api/v1/records/{rid}", headers=U,
                      params={"channels": "raw"}).status_code == 403
    full = client.get(f"/api/v1/records/{rid}", headers=REV).json()
    assert full["raw"][0]["raw"].startswith("{") and full["candidate"][0]["claims"]
    assert full["audit"]["trace"] == r["trace"] and full["audit"]["attempts"] == r["attempts"]
    assert full["user"]["citations"] == sorted({c for cl in r["claims"] if not cl["removed"]
                                                for c in cl["citations"]})
    other = {"Authorization": f"Bearer {token(sub='someone')}"}
    assert client.get(f"/api/v1/records/{rid}", headers=other).status_code == 403
    assert client.get("/api/v1/records", headers=U).json()[0]["request_id"] == rid


def test_withheld_answers_never_enter_chat_history(client):
    chat = client.post("/api/v1/chats", json={}, headers=U).json()
    cid = chat["chat_id"]
    r1 = client.post("/api/v1/query", json={"question": ESC, "chat_id": cid}, headers=U).json()
    assert r1["action"] == "ESCALATE"
    r2 = client.post("/api/v1/query", json={"question": Q, "chat_id": cid}, headers=U).json()
    assert r2["action"] in ("ANSWER", "ANSWER_WITH_CAUTION")
    full = client.get(f"/api/v1/chats/{cid}", headers=U).json()
    assert full["messages"][0]["answer"] == ""  # the escalation text is not stored as an answer
    assert full["history"][0]["withheld"] is True and "answer" not in full["history"][0]
    assert full["history"][1]["answer"] == r2["answer"] and full["history"][1]["claims"]
    rec = client.get(f"/api/v1/records/{r1['record_id']}", headers=U).json()
    assert rec["delivered"] is False and rec["history"]["withheld"]


def test_record_store_is_append_only(tmp_path):
    s = RecordStore(tmp_path / "r.sqlite")
    rec = {"request_id": "x1", "sub": "u", "chat_id": None, "action": "ANSWER", "delivered": True,
           "question": "q", "raw": [], "candidate": [], "user": {}, "audit": {}, "control": {},
           "history": {"question": "q", "answer": "a"}}  # fmt: skip
    s.save(rec)
    with pytest.raises(sqlite3.IntegrityError):
        s.save(rec)
    assert s.get("x1")["history"]["answer"] == "a"
    ctx = history_as_context([{"question": "q1", "answer": "a1"},
                              {"question": "q2", "withheld": True}])  # fmt: skip
    assert ctx == ["Q: q1 → delivered answer: a1"]
