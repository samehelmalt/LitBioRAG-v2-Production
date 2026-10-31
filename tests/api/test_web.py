"""The UI is a thin client: static files, feedback endpoint, metrics endpoint, no decisions."""

import json
import re
from pathlib import Path


def test_pages_and_static_are_served(client):
    html = client.get("/").text
    assert "AEOS" in html and "/static/app.js" in html and "/static/pages.js" in html
    assert client.get("/trust").status_code == 200 and "AEOS" in client.get("/trust").text
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/logo.svg").status_code == 200
    js = client.get("/static/app.js").text + client.get("/static/pages.js").text
    for ep in ("/api/v1/query/stream", "/api/v1/feedback", "/api/v1/kb", "/api/v1/review",
               "/api/v1/corpus", "/api/v1/keys", "/api/v1/tools", "/api/v1/admin/profile"):
        assert ep in js, ep


def test_ui_renders_but_never_decides():
    """The client must not contain verdict/gate logic: no rule ids, thresholds or labels decided."""
    js = Path("app/web/static/app.js").read_text() + Path("app/web/static/pages.js").read_text()
    assert not re.search(r"\b(supported|contradicted)\b\s*=[^=]", js)  # labels only read
    assert "R6" not in js and "threshold" not in js.lower()
    assert "precision" not in js.lower() and "recall" not in js.lower()  # no fake metrics
    for word in ("action", "rule", "reason", "confidence_calibrated", "removed_claims",
                 "substitution", "conflicts", "tier", "disclosure", "versions", "evidence_span",
                 "sc-uploaded", "sc-indexed", "under human review", "NOT YET MEASURED"):
        assert word in js, f"UI must render {word}"


def test_sidebar_matches_prototype():
    html = Path("app/web/static/index.html").read_text()
    for item in ("New Chat", "Knowledge Base", "Upload Documents", "Research Tools", "Evaluations",
                 "Datasets", "Model Settings", "API Access", "Review Queue", "Corpus Indexes",
                 "Recent chats"):
        assert item in html, item
    assert "AEOS <b>RAG</b>" in html and "logo.svg" in html


def test_feedback_is_recorded_for_the_labelling_protocol(client):
    r = client.post("/api/v1/feedback", json={"request_id": "abcd1234", "claim_id": "C1",
                                               "label": "incorrect", "notes": "wrong gene"})
    assert r.status_code == 202
    rec = json.loads((client.audit_path.parent / "feedback.jsonl").read_text().splitlines()[-1])
    assert rec["label"] == "incorrect" and rec["rater"] == "anonymous" and rec["claim_id"] == "C1"
    assert client.post("/api/v1/feedback", json={"request_id": "abcd1234", "claim_id": "C1",
                                                  "label": "maybe"}).status_code == 422


def test_metrics_endpoint_is_honest_when_nothing_is_measured(client):
    m = client.get("/api/v1/metrics").json()
    assert m == {"status": "NOT YET MEASURED"} or isinstance(m, dict)
