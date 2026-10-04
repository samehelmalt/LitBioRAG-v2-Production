import pytest

from app.evidence.selector import select_evidence
from app.reranking.base import LexicalOverlapReranker
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.encoders import HashEncoder
from app.retrieval.pipeline import MODES, HybridRetriever, RetrievalConfig
from app.retrieval.store import MetadataStore
from tests.retrieval.fixture_corpus import PLANTED_QUERIES


@pytest.fixture(scope="module")
def retriever(fixture_bundle):
    store = MetadataStore(fixture_bundle / "metadata.sqlite", read_only=True)
    cfg = RetrievalConfig(k_bm25=50, k_dense=50, rerank_pool=30, top_docs=10)
    return HybridRetriever(
        store,
        BM25Index(fixture_bundle / "bm25"),
        DenseIndex(fixture_bundle / "dense", HashEncoder(128)),
        LexicalOverlapReranker(),
        cfg,
    )


def test_config_from_yaml():
    cfg = RetrievalConfig.from_yaml("configs/retrieval.yaml")
    assert cfg.rrf_k == 60 and cfg.rerank_pool == 100 and cfg.top_docs == 20


@pytest.mark.parametrize("mode", MODES)
def test_planted_documents_rank_first_in_every_mode(retriever, mode):
    for question, gold, _ in PLANTED_QUERIES:
        res = retriever.retrieve(question, mode=mode)
        top = res.docs[0].doc_id
        assert top.endswith(gold[0]), (mode, question, top)
        assert res.audit.mode == mode and res.audit.timings_ms["total"] >= 0


def test_audit_ranks_cover_every_stage(retriever):
    res = retriever.retrieve("What does MAP3K3 do in NF-kappaB signalling?", mode="hybrid+rerank")
    ranks = res.audit.doc_ranks["pmid:900001"]
    assert set(ranks) == {"bm25", "dense", "fused", "reranked", "final"}
    assert ranks["final"] == 1
    assert res.audit.list_sizes["reranked"] <= 30
    # the near-miss gene is present but ranked below the true hit
    assert res.audit.rank_of("pmid:900002", "final") > 1


def test_mode_requirements(retriever, fixture_bundle):
    with pytest.raises(ValueError):
        retriever.retrieve("x", mode="magic")
    store = MetadataStore(fixture_bundle / "metadata.sqlite", read_only=True)
    bm25_only = HybridRetriever(store, BM25Index(fixture_bundle / "bm25"), None, None)
    assert bm25_only.retrieve("MAP3K3", mode="bm25").docs
    with pytest.raises(RuntimeError):
        bm25_only.retrieve("MAP3K3", mode="dense")
    with pytest.raises(RuntimeError):
        HybridRetriever(store, BM25Index(fixture_bundle / "bm25"),
                        DenseIndex(fixture_bundle / "dense", HashEncoder(128)), None
                        ).retrieve("MAP3K3", mode="hybrid+rerank")  # fmt: skip


def test_evidence_units_carry_provenance_and_flag_preprints(retriever):
    res = retriever.retrieve("Which mRNA properties does N6-methyladenosine (m6A) affect?")
    units = select_evidence(res, retriever.store, max_units=5, per_doc=2)
    assert units and units[0].doc_id == "doi:10.1101/2024.05.01.591234"
    u = units[0]
    assert u.citation_id == "DOI:10.1101/2024.05.01.591234"
    assert u.source_type.value == "preprint" and u.is_peer_reviewed is False
    doc = retriever.store.get_document(u.doc_id)
    assert doc.text[u.char_start : u.char_end] == u.text
    assert u.doc_rank == 1 and u.url.startswith("https://")
    pm = select_evidence(retriever.retrieve("MAP3K3 NF-kappaB"), retriever.store, max_units=3)
    assert pm[0].citation_id == "PMID:900001" and pm[0].is_peer_reviewed
    assert len({x.text[:120].lower() for x in pm}) == len(pm)  # de-duplicated
