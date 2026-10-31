"""Human review queue: enqueue on BLOCK, reviewer-only access, release with re-verification,
block, expiry, audit and label export."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.settings import ApiSettings, AuditSettings, AuthSettings, LimitSettings
from app.core.budget import UsageLedger
from app.review.service import ReviewConfig, ReviewService
from app.review.store import ReviewStore
from tests.api.conftest import SECRET, token

# a true, perfectly cited claim about the near neighbour: the gate blocks on R6 (substitution)
BLOCKED_Q = "What does MAP3K7 do in innate immunity?"  # fixture has MAP3K7 and MAP3K3 docs
ESCALATE_Q = "My mother takes warfarin 5 mg; what dose should she take tonight?"


@pytest.fixture
def app_env(pipeline, tmp_path, monkeypatch):
    monkeypatch.setenv("LITBIORAG_JWT_SECRET", SECRET)
    st = ApiSettings(bundle=str(pipeline.bundle), profile="test",
                     auth=AuthSettings(required=False),
                     limits=LimitSettings(rate_per_minute=600, burst=1000),
                     audit=AuditSettings(path=str(tmp_path / "audit.jsonl")))  # fmt: skip
    cfg = ReviewConfig(db_path=str(tmp_path / "r.sqlite"), labels_path=str(tmp_path / "l.jsonl"),
                       expire_hours=1)  # fmt: skip
    svc = ReviewService(pipeline, cfg, store=ReviewStore(tmp_path / "r.sqlite"),
                        labels_path=tmp_path / "labels.jsonl")  # fmt: skip
    with TestClient(create_app(st, pipeline=pipeline, ledger=UsageLedger(None), review=svc)) as c:
        yield c, svc, tmp_path


def _block_a_case(c):
    r = c.post("/api/v1/query", json={"question": ESCALATE_Q},
               headers={"Authorization": f"Bearer {token(sub='asker-1')}"})  # fmt: skip
    assert r.status_code == 200
    body = r.json()
    assert body["action"] == "ESCALATE" and body["review"]["status"] == "under_review"
    return body


def test_blocked_answer_is_enqueued_and_visible_to_asker(app_env):
    c, svc, _ = app_env
    body = _block_a_case(c)
    case_id = body["review"]["case_id"]
    assert svc.store.get(case_id)["status"] == "pending"
    mine = c.get("/api/v1/my/cases", headers={"Authorization": f"Bearer {token(sub='asker-1')}"})
    assert mine.status_code == 200 and mine.json()[0]["case_id"] == case_id
    assert mine.json()[0]["response"] is None  # the asker does not get the reviewer view
    assert c.get("/api/v1/my/cases").status_code == 401  # anonymous


def test_delivered_answers_are_not_enqueued(app_env):
    c, svc, _ = app_env
    r = c.post("/api/v1/query", json={"question": "What does MAP3K3 do in NF-kappaB signalling?"})
    assert r.json()["action"] in ("ANSWER", "ANSWER_WITH_CAUTION") and r.json()["review"] is None
    assert svc.store.counts()["pending"] == 0


def test_queue_requires_reviewer_role(app_env):
    c, _, _ = app_env
    _block_a_case(c)
    assert c.get("/api/v1/review/queue").status_code == 403
    hdr = {"Authorization": f"Bearer {token(sub='student')}"}
    assert c.get("/api/v1/review/queue", headers=hdr).status_code == 403
    rev = {"Authorization": f"Bearer {token(sub='dr-rev', roles=['reviewer'])}"}
    q = c.get("/api/v1/review/queue", headers=rev)
    assert q.status_code == 200 and len(q.json()) == 1 and q.json()[0]["status"] == "pending"
    full = c.get(f"/api/v1/review/{q.json()[0]['case_id']}", headers=rev).json()
    assert full["response"]["trace"] and full["response"]["action"] == "ESCALATE"


def test_release_with_edit_is_reverified_and_exported(app_env):
    c, svc, tmp = app_env
    case_id = _block_a_case(c)["review"]["case_id"]
    rev = {"Authorization": f"Bearer {token(sub='dr-rev', roles=['reviewer'])}"}
    # text not supported by the case evidence → 409 with the re-verification, nothing released
    r = c.post(f"/api/v1/review/{case_id}/decision", headers=rev,
               json={"decision": "release", "edited_answer": "Warfarin dosing must be set by "
                     "your clinician based on the INR; this tool cannot advise a dose."})
    assert r.status_code == 409 and r.json()["reverification"]["flagged_claims"]
    assert svc.store.get(case_id)["status"] == "pending"
    # reviewer confirms with the flags in view → released, audited, labelled
    r = c.post(f"/api/v1/review/{case_id}/decision", headers=rev,
               json={"decision": "release", "edited_answer": "Warfarin dosing must be set by "
                     "your clinician based on the INR; this tool cannot advise a dose.",
                     "notes": "generic safety guidance", "confirm": True})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["status"] == "released" and out["human_reviewed"]["reviewer"] == "dr-rev"
    assert out["reverification"]["clean"] is False
    mine = c.get("/api/v1/my/cases", headers={"Authorization": f"Bearer {token(sub='asker-1')}"})
    assert mine.json()[0]["released_answer"].startswith("Warfarin dosing must")
    label = json.loads((tmp / "labels.jsonl").read_text().splitlines()[-1])
    assert label["label"] == "acceptable" and label["rater"] == "dr-rev" and label["edited"]
    assert label["machine_action"] == "ESCALATE"
    # label shape is importable by the evaluation-side protocol
    from evaluation.labels import LabelRecord

    LabelRecord(**{k: label[k] for k in ("query_id", "question", "answer", "evidence", "label",
                                         "rater", "notes", "run_id", "proxy_label")})
    audit = [json.loads(line) for line in (tmp / "audit.jsonl").read_text().splitlines()]
    assert {a["event"] for a in audit} >= {"query", "review_enqueued", "review_decision"}
    # a decided case cannot be decided again
    r = c.post(f"/api/v1/review/{case_id}/decision", headers=rev, json={"decision": "block"})
    assert r.status_code == 409


def test_block_keeps_abstention_and_labels_unacceptable(app_env):
    c, svc, tmp = app_env
    case_id = _block_a_case(c)["review"]["case_id"]
    rev = {"Authorization": f"Bearer {token(sub='admin-1', roles=['admin'])}"}
    r = c.post(f"/api/v1/review/{case_id}/decision", headers=rev,
               json={"decision": "block", "notes": "out of scope, correctly stopped"})
    assert r.status_code == 200 and r.json()["status"] == "blocked"
    assert r.json()["released_answer"] is None and r.json()["human_reviewed"] is None
    label = json.loads((tmp / "labels.jsonl").read_text().splitlines()[-1])
    assert label["label"] == "unacceptable"
    assert c.get("/api/v1/review/stats", headers=rev).json()["blocked"] == 1


def test_release_from_kept_claims_is_clean_when_claims_were_supported(pipeline, tmp_path):
    """A release built from the machine's own supported claims re-verifies clean."""
    svc = ReviewService(pipeline, ReviewConfig(expire_hours=1),
                        store=ReviewStore(None), labels_path=tmp_path / "l.jsonl")  # fmt: skip
    out = pipeline.run("What does MAP3K3 do in NF-kappaB signalling?")
    from app.api.main import to_response

    resp = to_response(out, "req1", pipeline.versions.as_dict(), False).model_dump(mode="json")
    case = svc.store.create(request_id="req1", sub="u", question=out.question, course_ids=None,
                            action="ABSTAIN", internal_state="BLOCK", rule="R9-sufficiency",
                            reason="test", response=resp,
                            evidence=[u.model_dump(mode="json") for u in out.evidence])  # fmt: skip
    kept = [cl["id"] for cl in resp["claims"] if cl["label"] == "supported" and not cl["removed"]]
    done = svc.decide(case["case_id"], decision="release", reviewer="r", kept_claim_ids=kept)
    assert done["status"] == "released" and done["reverification"]["clean"]


def test_expiry(tmp_path):
    store = ReviewStore(tmp_path / "x.sqlite")
    case = store.create(request_id="r", sub="s", question="q", course_ids=None, action="ABSTAIN",
                        internal_state="BLOCK", rule="R3", reason="x", response={}, evidence=[])
    assert store.expire(max_age_s=3600) == 0
    assert store.expire(max_age_s=0, now=time.time() + 1) == 1
    assert store.get(case["case_id"])["status"] == "expired"
    with pytest.raises(LookupError):
        store.decide(case["case_id"], status="blocked", reviewer="r", notes="",
                     released_answer=None, released_claim_ids=None,
                     reverification=None)  # fmt: skip


def test_skip_reasons_are_not_enqueued(pipeline):
    svc = ReviewService(pipeline, ReviewConfig(), store=ReviewStore(None), labels_path=None)
    assert not svc.should_enqueue("ABSTAIN", "BLOCK", "no_evidence")
    assert svc.should_enqueue("ABSTAIN", "BLOCK", "insufficient_evidence")
    assert not svc.should_enqueue("ANSWER_WITH_CAUTION", "WARN", None)
    assert svc.should_enqueue("ESCALATE", "BLOCK", "out_of_scope")
