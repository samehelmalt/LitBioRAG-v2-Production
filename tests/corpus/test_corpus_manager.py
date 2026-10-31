"""Corpus index management: list/verify/activate/update bundles, indexed switch, admin-only."""

import json
import shutil
import time

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.settings import ApiSettings, AuditSettings, AuthSettings, LimitSettings
from app.core.budget import UsageLedger
from app.core.documents import Document, SourceType
from app.corpus.manager import CorpusConfig, CorpusManager
from app.service.pipeline import AnswerPipeline
from scripts.update_bundle import update_bundle
from tests.api.conftest import SECRET, token

ADMIN = {"Authorization": f"Bearer {token(sub='admin-1', roles=['admin'])}"}
USER = {"Authorization": f"Bearer {token(sub='u-1')}"}


def _wait(c, job_id, headers=ADMIN, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/v1/corpus/jobs/{job_id}", headers=headers).json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


@pytest.fixture
def env(fixture_bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("LITBIORAG_JWT_SECRET", SECRET)
    arts = tmp_path / "artifacts"
    arts.mkdir()
    shutil.copytree(fixture_bundle, arts / "fixture_v1")
    shutil.copytree(fixture_bundle, arts / "fixture_v2")
    man = json.loads((arts / "fixture_v2" / "manifest.json").read_text())
    man["corpus_id"] = "fixture_v2"
    (arts / "fixture_v2" / "manifest.json").write_text(json.dumps(man))
    pl = AnswerPipeline(arts / "fixture_v1", profile="test", ledger=UsageLedger(None),
                        kb_root=tmp_path / "kb")  # fmt: skip
    st = ApiSettings(bundle=str(arts / "fixture_v1"), profile="test",
                     auth=AuthSettings(required=False),
                     limits=LimitSettings(rate_per_minute=600, burst=1000),
                     audit=AuditSettings(path=str(tmp_path / "audit.jsonl")))  # fmt: skip
    cfg = CorpusConfig(artifacts_dir=str(arts), jobs_dir=str(tmp_path / "jobs"),
                       staging_dir=str(tmp_path / "staging"), allow_build_on_serve=False,
                       encoder_spec="hash:128")  # fmt: skip
    with TestClient(create_app(st, pipeline=pl, ledger=UsageLedger(None), corpus_config=cfg)) as c:
        c.arts = arts
        c.pipeline = pl
        yield c


def test_admin_only_and_listing(env):
    assert env.get("/api/v1/corpus").status_code == 403
    assert env.get("/api/v1/corpus", headers=USER).status_code == 403
    s = env.get("/api/v1/corpus", headers=ADMIN).json()
    ids = {b["corpus_id"]: b for b in s["bundles"]}
    assert set(ids) == {"fixture_v1", "fixture_v2"} and ids["fixture_v1"]["active"]
    assert ids["fixture_v1"]["encoder"] == "hash:128"
    assert ids["fixture_v1"]["encoder_matches_profile"]
    assert s["indexed_enabled"] and "build host" in s["rebuild_hint"]


def test_verify_and_activate(env):
    r = env.post("/api/v1/corpus/jobs", headers=ADMIN,
                 json={"type": "verify", "params": {"corpus_id": "fixture_v2"}})  # fmt: skip
    assert r.status_code == 202
    j = _wait(env, r.json()["job_id"])
    assert j["status"] == "done" and j["result"]["ok"] and j["log_tail"]
    r = env.post("/api/v1/corpus/activate", json={"corpus_id": "fixture_v2"}, headers=ADMIN)
    assert r.status_code == 200 and r.json()["active"] == "fixture_v2"
    assert env.get("/health").json()["bundle_id"] == "fixture_v2"
    assert env.post("/api/v1/corpus/activate", json={"corpus_id": "nope"},
                    headers=ADMIN).status_code == 404
    # a corrupted bundle is refused
    (env.arts / "fixture_v1" / "metadata.sqlite").write_bytes(b"corrupt")
    assert env.post("/api/v1/corpus/activate", json={"corpus_id": "fixture_v1"},
                    headers=ADMIN).status_code == 409


def test_update_job_appends_documents_and_reloads(env, tmp_path):
    new = tmp_path / "new.jsonl"
    doc = Document(doc_id="pmid:777001", pmid="777001", year=2026,
                   title="Semaglutide and weight loss in adolescents",
                   abstract="Semaglutide 2.4 mg weekly reduced BMI by 16% over 68 weeks "
                            "in adolescents with obesity.",
                   source_type=SourceType.RCT)  # fmt: skip
    new.write_text(doc.model_dump_json() + "\n")
    before = env.post("/api/v1/query", json={"question": "Does semaglutide reduce BMI in "
                                                         "adolescents?", "scope": "indexed"}).json()
    assert not any(s["citation_id"] == "PMID:777001" for s in before["sources"])
    r = env.post("/api/v1/corpus/jobs", headers=ADMIN,
                 json={"type": "update", "params": {"corpus_id": "fixture_v1", "docs": [str(new)]}})
    j = _wait(env, r.json()["job_id"])
    assert j["status"] == "done", j
    assert j["result"]["added_documents"] == 1 and j["result"]["added_passages"] >= 1
    after = env.post("/api/v1/query", json={"question": "Does semaglutide reduce BMI in "
                                                        "adolescents?", "scope": "indexed"}).json()
    assert after["sources"][0]["citation_id"] == "PMID:777001"
    man = json.loads((env.arts / "fixture_v1" / "manifest.json").read_text())
    assert man["updates"][-1]["documents"] == 1 and man["counts"]["documents"] == 61
    # re-updating the same document replaces it (no duplicate), stale vectors are counted
    res = update_bundle(env.arts / "fixture_v1", [str(new)], "hash:128")
    assert res["replaced"] == 1 and res["stale_vectors"] >= 1
    with pytest.raises(ValueError):
        update_bundle(env.arts / "fixture_v1", [str(new)], "hash:64")  # wrong encoder refused


def test_rebuild_refused_on_serving_host_with_hint(env):
    r = env.post("/api/v1/corpus/jobs", headers=ADMIN,
                 json={"type": "rebuild", "params": {"corpus_id": "x", "docs": ["a.jsonl"]}})
    assert r.status_code == 409 and "scripts.build_indexes" in r.json()["error"]
    assert env.post("/api/v1/corpus/jobs", headers=ADMIN,
                    json={"type": "verify", "params": {}}).status_code in (202, 422)


def test_indexed_disabled_forces_uploaded_scope(env):
    assert env.post("/api/v1/corpus/enabled", json={"enabled": False}, headers=ADMIN).json() == {
        "indexed_enabled": False}
    r = env.post("/api/v1/query", json={"question": "What does MAP3K3 do?", "scope": "indexed"},
                 headers=USER)
    assert r.status_code == 422
    r = env.post("/api/v1/query", json={"question": "What does MAP3K3 do?"}, headers=USER)
    assert r.status_code == 422  # default scope 'both' → uploaded, and the user has no KB yet
    kb = env.post("/api/v1/kb", json={"name": "mine"}, headers=USER).json()
    md = env.pipeline.kb.root / "x.md"
    md.write_text("# MAP3K3\nMAP3K3 (MEKK3) activates the IKK complex upstream of NF-kappaB.\n")
    with open(md, "rb") as fh:
        env.post(f"/api/v1/kb/{kb['kb_id']}/upload", headers=USER,
                 files={"file": ("x.md", fh, "text/markdown")})
    r = env.post("/api/v1/query", json={"question": "What does MAP3K3 do?"}, headers=USER).json()
    assert r["scope"] == "uploaded" and all(s["citation_id"].startswith("FILE:")
                                            for s in r["sources"])
    env.post("/api/v1/corpus/enabled", json={"enabled": True}, headers=ADMIN)
    r = env.post("/api/v1/query", json={"question": "What does MAP3K3 do?"}, headers=USER).json()
    assert r["scope"] == "both" and any(s["citation_id"] == "PMID:900001" for s in r["sources"])


def test_manager_without_api(tmp_path, fixture_bundle):
    m = CorpusManager(CorpusConfig(artifacts_dir=str(tmp_path / "a"), jobs_dir=str(tmp_path / "j"),
                                   staging_dir=str(tmp_path / "s")), active_bundle=fixture_bundle)
    assert m.list_bundles()[0]["active"] is True
    with pytest.raises(ValueError):
        m.start_job("nope", {})
    with pytest.raises(PermissionError):
        m.start_job("rebuild", {"corpus_id": "c", "docs": ["d.jsonl"]})
