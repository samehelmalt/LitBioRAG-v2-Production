"""Follow-ups are rewritten and shown; widget and embed page served; tenants are isolated."""

from tests.api.conftest import token

U = {"Authorization": f"Bearer {token(sub='u-f')}"}


def test_follow_up_is_interpreted_and_shown(client):
    chat = client.post("/api/v1/chats", json={}, headers=U).json()
    r1 = client.post("/api/v1/query", json={"question": "What does MAP3K3 do in NF-kappaB "
                                                        "signalling?", "chat_id": chat["chat_id"]},
                     headers=U).json()
    assert r1["interpreted_as"] is None
    r2 = client.post("/api/v1/query", json={"question": "Why is that?",
                                            "chat_id": chat["chat_id"]}, headers=U).json()
    assert r2["interpreted_as"] and "map3k3" in r2["interpreted_as"]
    assert "not evidence" in r2["interpretation_note"]
    rec = client.get(f"/api/v1/records/{r2['record_id']}", headers=U).json()
    assert rec["control"]["interpreted_as"] == r2["interpreted_as"]
    assert rec["history"]["question"] == "Why is that?"  # the asked question is kept verbatim


def test_span_in_prompt_is_reported(client):
    r = client.post("/api/v1/query", json={"question": "What does MAP3K3 do in NF-kappaB "
                                                       "signalling?"}, headers=U).json()
    assert "span_in_prompt_rate" in r["verification"]
    assert r["verification"]["span_in_prompt_rate"] in (None, 1.0)


def test_widget_and_embed_are_served(client):
    assert client.get("/embed").status_code == 200
    js = client.get("/static/widget.js").text
    assert "/api/v1/query" in js and "X-API-Key" in js and "not medical advice" in js


def test_tenants_never_share_knowledge_bases_or_chats(kb_client):  # noqa: F811
    c = kb_client
    import time

    import jwt

    from tests.api.conftest import SECRET

    def tok(tenant, sub="alice"):
        return {"Authorization": "Bearer " + jwt.encode(
            {"sub": sub, "tenant": tenant, "roles": [], "courses": [], "aud": "litbiorag",
             "exp": int(time.time()) + 600}, SECRET, algorithm="HS256")}

    a, b = tok("clinic-a"), tok("clinic-b")
    kb = c.post("/api/v1/kb", json={"name": "A's notes"}, headers=a).json()
    assert kb["owner"] == "clinic-a/alice"
    assert c.get("/api/v1/kb", headers=b).json() == []  # same sub, other tenant: nothing
    assert c.get(f"/api/v1/kb/{kb['kb_id']}", headers=b).status_code == 403
    chat = c.post("/api/v1/chats", json={}, headers=a).json()
    assert c.get(f"/api/v1/chats/{chat['chat_id']}", headers=b).status_code == 404
    assert c.post("/api/v1/query", json={"question": "x y z", "scope": "uploaded",
                                         "kb_ids": [kb["kb_id"]]}, headers=b).status_code == 403
