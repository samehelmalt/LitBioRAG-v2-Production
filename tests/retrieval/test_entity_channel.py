from app.reranking.base import LexicalOverlapReranker
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.encoders import HashEncoder
from app.retrieval.pipeline import HybridRetriever, RetrievalConfig
from app.retrieval.store import MetadataStore


def test_exact_symbol_channel_only_matches_the_asked_gene(fixture_bundle):
    bm25 = BM25Index(fixture_bundle / "bm25")
    hits = bm25.search_must(["MAP3K3"], k=20)
    docs = {h.doc_id for h in hits}
    assert "pmid:900001" in docs and "pmid:900002" in docs  # 900002 mentions MAP3K3 explicitly
    assert bm25.search_must(["MAP3K9"], k=20) == []  # near neighbour never matches
    assert bm25.search_must(["TAS-102"], k=20)[0].doc_id == "pmid:900003"

    store = MetadataStore(fixture_bundle / "metadata.sqlite", read_only=True)
    cfg = RetrievalConfig(k_bm25=50, k_dense=50, rerank_pool=30, top_docs=10, entity_boost=1.0)
    r = HybridRetriever(store, bm25, DenseIndex(fixture_bundle / "dense", HashEncoder(128)),
                        LexicalOverlapReranker(), cfg)  # fmt: skip
    res = r.retrieve("What does MAP3K3 do in NF-kappaB signalling?", mode="hybrid+rerank")
    assert res.audit.query_symbols == ["MAP3K3"]
    assert "entity" in res.audit.doc_ranks["pmid:900001"] and res.docs[0].doc_id == "pmid:900001"
    assert "entity" in res.audit.list_sizes and res.audit.timings_ms["entity"] >= 0

    off = RetrievalConfig(k_bm25=50, k_dense=50, rerank_pool=30, top_docs=10, entity_boost=0.0)
    dense = DenseIndex(fixture_bundle / "dense", HashEncoder(128))
    res_off = HybridRetriever(store, bm25, dense, LexicalOverlapReranker(), off).retrieve(
        "MAP3K3 NF-kappaB"
    )
    assert "entity" not in res_off.audit.list_sizes
