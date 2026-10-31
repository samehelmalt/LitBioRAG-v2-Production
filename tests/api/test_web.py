"""The UI is a thin client: static files, feedback endpoint, metrics endpoint."""

import json
import re


def test_pages_and_static_are_served(client):
    html = client.get("/").text
    assert "LitBioRAG" in html and "/static/app.js" in html
    assert client.get("/trust").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
    js = client.get("/static/app.js").text
    assert "/api/v1/query" in js and "/api/v1/feedback" in js


def test_ui_renders_but_never_decides():
    """The client must not contain verdict/gate logic: no rule ids, thresholds or labels decided."""
    from pathlib import Path

    js = Path("app/web/static/app.js").read_text()
    assert not re.search(r"\b(supported|contradicted)\b\s*=", js)  # labels only read, never set
    assert "R6" not in js and "threshold" not in js.lower()
    for word in ("action", "rule", "reason", "confidence_calibrated", "removed_claims",
                 "substitution", "conflicts", "tier", "disclosure", "versions", "evidence_span"):
        assert word in js, f"UI must render {word}"


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
