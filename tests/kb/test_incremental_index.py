"""Incremental KB index: add/delete cycles keep retrieval correct without full rebuilds; the
index survives a restart; concurrent queries never read a half-written index; jobs persist."""

import threading
from pathlib import Path

import pytest

pytest.importorskip("faiss")
pytest.importorskip("tantivy")

from app.kb.dense_store import IncrementalDense  # noqa: E402
from app.kb.indexer import KBIndexer  # noqa: E402
from app.kb.registry import KBRegistry  # noqa: E402
from app.retrieval.encoders import HashEncoder  # noqa: E402
from app.retrieval.pipeline import HybridRetriever, RetrievalConfig  # noqa: E402


def _md(tmp: Path, name: str, lines: list[str]) -> Path:
    p = tmp / name
    p.write_text("\n".join(f"# {name} part {i}\n{line}" for i, line in enumerate(lines)))
    return p


def test_incremental_dense_add_remove_persist(tmp_path):
    enc = HashEncoder(32)
    d = IncrementalDense(tmp_path / "dense", enc)
    d.add(["a#0", "b#0"], ["atorvastatin lowers ldl", "hyperkalemia peaked t waves"])
    assert d.ntotal == 2 and d.search("peaked t waves", k=1)[0].passage_id == "b#0"
    d.add(["a#0"], ["atorvastatin lowers ldl cholesterol strongly"])  # re-add replaces
    assert d.ntotal == 2
    assert d.remove(["b#0"]) == 1 and d.ntotal == 1
    assert [h.passage_id for h in d.search("peaked t waves", k=5)] == ["a#0"]
    again = IncrementalDense(tmp_path / "dense", enc)  # restart: loads the persisted state
    assert again.ntotal == 1 and again.search("ldl", k=1)[0].passage_id == "a#0"
    other = IncrementalDense(tmp_path / "dense", HashEncoder(64))  # encoder change → empty
    assert other.ntotal == 0


def test_indexer_add_delete_cycles_without_rebuild(tmp_path):
    ix = KBIndexer(tmp_path / "kb", HashEncoder(64), registry=KBRegistry(None))
    kb = ix.registry.create(name="k", owner="u")
    rebuilds = {"n": 0}
    orig = ix._rebuild_dense

    def counting():
        rebuilds["n"] += 1
        orig()

    ix._rebuild_dense = counting
    files = []
    for i in range(6):
        f = _md(tmp_path, f"doc{i}.md", [f"Drug{i} reduces outcome{i} by {10 + i} percent."] * 3)
        files.append(ix.add_file(kb["kb_id"], f, file_name=f.name, uploaded_by="u")["file_id"])
    assert ix.store.counts()["documents"] == 18 and ix.dense.ntotal == 18
    r = HybridRetriever(ix.store, ix.bm25, ix.dense, None, RetrievalConfig())
    assert r.retrieve("Drug3 outcome3", mode="hybrid").docs[0].doc_id.startswith(f"file:{files[3]}")
    ix.remove_file(kb["kb_id"], files[3])
    assert ix.store.counts()["documents"] == 15 and ix.dense.ntotal == 15
    res = r.retrieve("Drug3 outcome3", mode="hybrid")
    assert not any(d.doc_id.startswith(f"file:{files[3]}") for d in res.docs)
    assert rebuilds["n"] == 0  # no full rebuild was needed
    assert ix.jobs[f"{kb['kb_id']}:{files[0]}"]["status"] == "done"


def test_jobs_and_index_survive_restart(tmp_path):
    root = tmp_path / "kb"
    ix = KBIndexer(root, HashEncoder(32))
    kb = ix.registry.create(name="k", owner="u")
    f = _md(tmp_path, "a.md", ["Metformin lowers HbA1c by about one percent."])
    fid = ix.add_file(kb["kb_id"], f, file_name="a.md", uploaded_by="u")["file_id"]
    ix2 = KBIndexer(root, HashEncoder(32))  # same encoder: nothing re-embedded
    assert ix2.dense.ntotal == 1 and ix2.jobs[f"{kb['kb_id']}:{fid}"]["status"] == "done"
    ix3 = KBIndexer(root, HashEncoder(48))  # encoder changed: re-embedded from the store
    assert ix3.dense.ntotal == 1 and ix3.dense.encoder.dim == 48


def test_concurrent_queries_during_uploads(tmp_path):
    ix = KBIndexer(tmp_path / "kb", HashEncoder(32), registry=KBRegistry(None))
    kb = ix.registry.create(name="k", owner="u")
    f0 = _md(tmp_path, "seed.md", ["Seed document about aspirin and platelets."])
    ix.add_file(kb["kb_id"], f0, file_name="seed.md", uploaded_by="u")
    errors: list[Exception] = []
    stop = threading.Event()

    def reader():
        r = HybridRetriever(ix.store, ix.bm25, ix.dense, None, RetrievalConfig())
        while not stop.is_set():
            try:
                r.retrieve("aspirin platelets", mode="hybrid")
            except Exception as e:  # noqa: BLE001
                errors.append(e)
                return

    t = threading.Thread(target=reader)
    t.start()
    for i in range(8):
        f = _md(tmp_path, f"u{i}.md", [f"Upload {i} about warfarin{i} and INR monitoring."])
        ix.add_file(kb["kb_id"], f, file_name=f.name, uploaded_by="u")
    stop.set()
    t.join(timeout=10)
    assert not errors and ix.dense.ntotal == 9
