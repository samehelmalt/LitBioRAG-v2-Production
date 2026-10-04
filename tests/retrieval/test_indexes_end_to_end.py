"""Build a bundle from a 60-document fixture and check both indexes find planted documents."""

import json

import pytest

pytest.importorskip("tantivy")
pytest.importorskip("faiss")

from app.core.documents import Document, SourceType  # noqa: E402
from app.retrieval.bm25 import BM25Index, normalize_for_bm25  # noqa: E402
from app.retrieval.dense import DenseIndex  # noqa: E402
from app.retrieval.encoders import HashEncoder  # noqa: E402
from app.retrieval.store import MetadataStore  # noqa: E402
from scripts import artifacts, build_indexes  # noqa: E402

PLANTED = {
    "pmid:900001": (
        "MAP3K3 regulates NF-kappaB activation in endothelial cells",
        "We show that MAP3K3 (MEKK3) is required for NF-kappaB signalling. Knockdown of MAP3K3 "
        "reduced p65 nuclear translocation by 61%.",
    ),
    "pmid:900002": (
        "MAP3K7 (TAK1) in innate immunity",
        "MAP3K7, also known as TAK1, mediates TLR signalling. This study does not examine MAP3K3.",
    ),
    "pmid:900003": (
        "Trifluridine/tipiracil versus fluoropyrimidines in colorectal cancer",
        "Trifluridine/tipiracil (TAS-102) retained activity in 5-FU refractory tumours because "
        "trifluridine is incorporated into DNA rather than inhibiting thymidylate synthase.",
    ),
    "doi:10.1101/2024.05.01.591234": (
        "N6-methyladenosine (m6A) controls mRNA stability",
        "m6A marks on mRNA alter stability, splicing and translation efficiency.",
    ),
}


def _docs():
    for doc_id, (title, abstract) in PLANTED.items():
        if doc_id.startswith("pmid:"):
            yield Document(doc_id=doc_id, pmid=doc_id[5:], title=title, abstract=abstract,
                           source_type=SourceType.JOURNAL_ARTICLE, year=2016)  # fmt: skip
        else:
            yield Document(doc_id=doc_id, doi=doc_id[4:], title=title, abstract=abstract,
                           source_type=SourceType.PREPRINT, year=2024,
                           source="biorxiv")  # fmt: skip
    for i in range(56):  # filler on unrelated topics
        yield Document(
            doc_id=f"pmid:{100 + i}", pmid=str(100 + i), year=2015 + i % 10,
            title=f"Study {i} of hip fracture rehabilitation outcomes",
            abstract=f"Cohort {i}: physiotherapy after hip fracture improved walking distance "
            f"by {10 + i} metres at six weeks. Adherence was {50 + i} percent.",
            source_type=SourceType.OBSERVATIONAL,
        )  # fmt: skip


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("bundle")
    docs = root / "docs.jsonl"
    with open(docs, "w", encoding="utf-8") as fh:
        for d in _docs():
            fh.write(d.model_dump_json() + "\n")
    out = root / "artifact"
    rc = build_indexes.main(
        ["--docs", str(docs), "--out", str(out), "--corpus-id", "fixture_v1",
         "--encoder", "hash:128", "--shard-size", "25", "--batch-size", "8"]  # fmt: skip
    )
    assert rc == 0
    return out


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
