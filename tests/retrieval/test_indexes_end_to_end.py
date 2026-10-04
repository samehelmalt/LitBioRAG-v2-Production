"""Build a bundle from a 60-document fixture and check both indexes find planted documents."""

import json

import pytest

pytest.importorskip("tantivy")
pytest.importorskip("faiss")

from app.retrieval.bm25 import BM25Index, normalize_for_bm25  # noqa: E402
from app.retrieval.dense import DenseIndex  # noqa: E402
from app.retrieval.encoders import HashEncoder  # noqa: E402
from app.retrieval.store import MetadataStore  # noqa: E402
from scripts import artifacts, build_indexes  # noqa: E402


@pytest.fixture(scope="module")
def bundle(fixture_bundle):
    """The shared session bundle from tests/retrieval/conftest.py."""
    return fixture_bundle


def test_bundle_contents_and_manifest(bundle):
    assert (bundle / "metadata.sqlite").exists()
    assert (bundle / "bm25").is_dir() and (bundle / "dense" / "index.faiss").exists()
    m = json.loads((bundle / "manifest.json").read_text())
    assert m["corpus_id"] == "fixture_v1" and m["counts"]["documents"] == 60
    assert m["encoder"]["encoder"] == "hash:128" and m["dense"]["type"] == "FlatIP"
    assert artifacts.verify(bundle) == []
    assert not list((bundle / "dense" / "shards").glob("*.npy"))  # shards removed after build


def test_rerun_is_idempotent(bundle, capsys):
    rc = build_indexes.main(
        ["--docs", "nonexistent*.jsonl", "--out", str(bundle), "--corpus-id", "fixture_v1",
         "--encoder", "hash:128"]  # fmt: skip
    )
    assert rc == 0
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    stages = ("store", "bm25", "embed", "dense")
    assert all(entry.get("skipped") for entry in logs if entry["stage"] in stages)


def test_bm25_exact_gene_symbol_and_split_tokens(bundle):
    idx = BM25Index(bundle / "bm25")
    assert idx.num_docs() >= 60
    top = [h.doc_id for h in idx.search("What does MAP3K3 do in NF-kappaB signalling?", k=3)]
    assert top[0] == "pmid:900001"
    norm = normalize_for_bm25("Trifluridine/Tipiracil (TAS-102). Dose 2.5 mg.")
    assert norm == "trifluridine tipiracil tas 102 dose 2.5 mg"
    q = "Why is Trifluridine/Tipiracil active in colorectal cancer?"
    top = [h.doc_id for h in idx.search(q, k=3)]
    assert top[0] == "pmid:900003"
    top = [h.doc_id for h in idx.search("N6-methyladenosine (m6A) mRNA properties", k=3)]
    assert top[0] == "doi:10.1101/2024.05.01.591234"
    assert idx.search("!!! ::: ((( )))") == []


def test_dense_search_and_store_resolution(bundle):
    dense = DenseIndex(bundle / "dense", HashEncoder(128))
    hits = dense.search("MAP3K3 NF-kappaB endothelial", k=5)
    assert hits[0].passage_id.startswith("pmid:900001#")
    store = MetadataStore(bundle / "metadata.sqlite", read_only=True)
    p = store.get_passage(hits[0].passage_id)
    assert p is not None and "MAP3K3" in p.text
    doc = store.get_document(p.doc_id)
    assert doc is not None and doc.text[p.char_start : p.char_end] == p.text
    with pytest.raises(ValueError):
        DenseIndex(bundle / "dense", HashEncoder(64))


def test_fetch_copies_and_verifies(bundle, tmp_path):
    dest = tmp_path / "served"
    assert artifacts.fetch(str(bundle), dest) == []
    # corrupt one file -> verify reports it; fetch repairs it
    target = dest / "dense" / "ids.npy"
    target.write_bytes(b"corrupt")
    assert any("ids.npy" in p for p in artifacts.verify(dest))
    assert artifacts.fetch(str(bundle), dest) == []
