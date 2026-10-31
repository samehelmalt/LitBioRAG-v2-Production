"""Upload → index → query with scope uploaded / indexed / both; entitlement; delete."""

from tests.api.conftest import token
from tests.kb.fixtures import make_docx, make_pdf, make_xlsx

OWNER = {"Authorization": f"Bearer {token(sub='owner-1')}"}
OTHER = {"Authorization": f"Bearer {token(sub='other-2')}"}
Q_KB = "Does dapagliflozin reduce heart failure admissions?"
Q_CORPUS = "What does MAP3K3 do in NF-kappaB signalling?"


def _upload(c, kb_id, path, headers=OWNER):
    with open(path, "rb") as fh:
        return c.post(f"/api/v1/kb/{kb_id}/upload", headers=headers,
                      files={"file": (path.name, fh, "application/octet-stream")})  # fmt: skip


def test_upload_index_query_and_entitlement(kb_client, tmp_path):
    c = kb_client
    assert c.post("/api/v1/kb", json={"name": "x"}).status_code == 401  # anonymous
    kb = c.post("/api/v1/kb", json={"name": "Cardio notes"}, headers=OWNER).json()
    kb_id = kb["kb_id"]
    pdf = make_pdf(tmp_path / "hf.pdf", [
        "Dapagliflozin reduced heart failure admissions by 30 percent in DAPA-HF.",
        "Empagliflozin showed a similar reduction in EMPEROR-Reduced."])
    r = _upload(c, kb_id, pdf)
    assert r.status_code == 202 and r.json()["status"] == "indexed" and r.json()["n_units"] == 2
    xlsx = make_xlsx(tmp_path / "doses.xlsx", {"Doses": [["Drug", "Dose"],
                                                           ["dapagliflozin", "10 mg daily"]]})
    assert _upload(c, kb_id, xlsx).status_code == 202
    status = c.get(f"/api/v1/kb/{kb_id}/status", headers=OWNER).json()
    assert status["version"] == 2 and len(status["files"]) == 2

    # scope uploaded: sources are FILE citations with locators
    r = c.post("/api/v1/query", json={"question": Q_KB, "scope": "uploaded"}, headers=OWNER)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scope"] == "uploaded" and body["kb_ids"] == [kb_id]
    assert body["sources"] and all(s["citation_id"].startswith("FILE:") for s in body["sources"])
    top = body["sources"][0]
    assert top["tier"] == "other" and top["locator"] == "p1" and top["locator_label"] == "page 1"
    assert top["display_source"] == "hf.pdf, page 1"
    assert any(cl["label"] == "supported" for cl in body["claims"])
    # the sources endpoint resolves uploaded units, with entitlement
    d = c.get(f"/api/v1/sources/{top['doc_id']}", headers=OWNER).json()
    assert d["file"]["kb_id"] == kb_id and d["citation_id"] == top["citation_id"]
    assert c.get(f"/api/v1/sources/{top['doc_id']}", headers=OTHER).status_code == 403

    # scope indexed: no uploaded sources at all
    r = c.post("/api/v1/query", json={"question": Q_KB, "scope": "indexed"}, headers=OWNER).json()
    assert not any(s["citation_id"].startswith("FILE:") for s in r["sources"])
    # scope both: corpus question still finds the planted PubMed document first
    r = c.post("/api/v1/query", json={"question": Q_CORPUS, "scope": "both"}, headers=OWNER).json()
    assert r["sources"][0]["citation_id"] == "PMID:900001"
    kinds = {s["citation_id"].split(":")[0] for s in r["sources"]}
    assert "PMID" in kinds

    # another user: cannot read the KB, cannot query it, has no uploaded scope
    assert c.get(f"/api/v1/kb/{kb_id}", headers=OTHER).status_code == 403
    assert c.post("/api/v1/query", json={"question": Q_KB, "scope": "uploaded",
                                         "kb_ids": [kb_id]}, headers=OTHER).status_code == 403
    assert c.post("/api/v1/query", json={"question": Q_KB, "scope": "uploaded"},
                  headers=OTHER).status_code == 422
    assert _upload(c, kb_id, pdf, headers=OTHER).status_code == 403
    # sharing grants read access
    assert c.post(f"/api/v1/kb/{kb_id}/share", json={"subs": ["other-2"]},
                  headers=OWNER).status_code == 200
    r = c.post("/api/v1/query", json={"question": Q_KB, "scope": "uploaded"}, headers=OTHER)
    assert r.status_code == 200 and r.json()["kb_ids"] == [kb_id]

    # delete a file → its units disappear from retrieval
    fid = status["files"][0]["file_id"]
    assert c.delete(f"/api/v1/kb/{kb_id}/files/{fid}", headers=OWNER).json()["removed_units"] == 2
    r = c.post("/api/v1/query", json={"question": Q_KB, "scope": "uploaded"}, headers=OWNER).json()
    assert not any(s["file_name"] == "hf.pdf" for s in r["sources"])
    assert c.delete(f"/api/v1/kb/{kb_id}", headers=OWNER).status_code == 200
    assert c.get("/api/v1/kb", headers=OWNER).json() == []


def test_upload_limits_and_types(kb_client, tmp_path):
    c = kb_client
    kb_id = c.post("/api/v1/kb", json={"name": "k"}, headers=OWNER).json()["kb_id"]
    bad = tmp_path / "x.exe"
    bad.write_bytes(b"MZ")
    assert _upload(c, kb_id, bad).status_code == 415
    big = tmp_path / "big.txt"
    big.write_bytes(b"a" * (1024 * 1024 + 10))
    assert _upload(c, kb_id, big).status_code == 413
    docx = make_docx(tmp_path / "n.docx", [("Dosing", ["Start 500 mg daily and titrate."])])
    r = _upload(c, kb_id, docx)
    assert r.status_code == 202 and r.json()["kind"] == "docx"
    # re-uploading the same content is idempotent (same file id, units replaced not duplicated)
    assert _upload(c, kb_id, docx).json()["file_id"] == r.json()["file_id"]
    assert c.pipeline.kb.store.counts()["documents"] == 1


def test_corrupt_file_is_reported_not_fatal(kb_client, tmp_path):
    c = kb_client
    kb_id = c.post("/api/v1/kb", json={"name": "k"}, headers=OWNER).json()["kb_id"]
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4 garbage")
    r = _upload(c, kb_id, broken)
    assert r.status_code == 202 and r.json()["status"] in ("failed", "indexed")
    st = c.get(f"/api/v1/kb/{kb_id}/status", headers=OWNER).json()
    assert st["files"][0]["status"] in ("failed", "indexed")
    assert c.get("/health").status_code == 200
