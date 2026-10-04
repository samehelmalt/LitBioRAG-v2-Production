import pytest

from app.reranking.base import Candidate, LexicalOverlapReranker
from app.retrieval.fusion import RankedItem, aggregate_to_documents, doc_id_of, rrf


def R(*ids):
    return [RankedItem(i, 1.0 / (n + 1)) for n, i in enumerate(ids)]


def test_rrf_hand_computed():
    fused = rrf({"bm25": R("a", "b", "c"), "dense": R("b", "d")}, k=60)
    by = {f.id: f for f in fused}
    assert by["b"].score == pytest.approx(1 / 62 + 1 / 61)
    assert by["a"].score == pytest.approx(1 / 61)
    assert by["d"].score == pytest.approx(1 / 62)
    assert [f.id for f in fused][:2] == ["b", "a"]
    assert by["b"].ranks == {"bm25": 2, "dense": 1} and by["c"].ranks == {"bm25": 3}


def test_rrf_weights_and_duplicates():
    fused = rrf({"bm25": R("a", "a", "b"), "dense": R("b")}, k=10, weights={"dense": 2.0})
    by = {f.id: f for f in fused}
    assert by["a"].score == pytest.approx(1 / 11)  # duplicate 'a' counted once, at rank 1
    assert by["b"].score == pytest.approx(1 / 13 + 2 / 11)
    with pytest.raises(ValueError):
        rrf({"x": R("a")}, k=0)


def test_aggregate_to_documents():
    items = [RankedItem("pmid:1#2", 0.9), RankedItem("pmid:2#0", 0.8), RankedItem("pmid:1#0", 0.7),
             RankedItem("pmid:1#1", 0.6), RankedItem("doi:10.1/x#0", 0.5)]  # fmt: skip
    docs = aggregate_to_documents(items, max_passages_per_doc=2)
    assert [d.doc_id for d in docs] == ["pmid:1", "pmid:2", "doi:10.1/x"]
    assert docs[0].passage_ids == ["pmid:1#2", "pmid:1#0"] and docs[0].score == 0.9
    assert doc_id_of("doi:10.1101/2024.05.01.591234#3") == "doi:10.1101/2024.05.01.591234"


def test_lexical_reranker_prefers_specific_matches():
    cands = [
        Candidate("p1", "MAP3K3 is required for NF-kappaB signalling in endothelial cells."),
        Candidate("p2", "MAP3K7 mediates TLR signalling in innate immunity."),
        Candidate("p3", "Hip fracture rehabilitation improves walking distance."),
    ]
    hits = LexicalOverlapReranker().rerank("What does MAP3K3 do in NF-kappaB signalling?", cands, 2)
    assert [h.passage_id for h in hits] == ["p1", "p2"]
    assert LexicalOverlapReranker().rerank("the of", cands, 2) == []
