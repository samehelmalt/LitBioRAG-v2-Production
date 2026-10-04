import math

import pytest

from evaluation.metrics.retrieval import (
    first_relevant_rank,
    mrr,
    ndcg_at_k,
    ranking_metrics,
    recall_at_k,
)

RANKED = ["a", "b", "c", "d", "e"]


def test_recall_and_mrr():
    assert recall_at_k(RANKED, ["c", "z"], k=5) == 0.5
    assert recall_at_k(RANKED, ["c"], k=2) == 0.0
    assert mrr(RANKED, ["c"]) == pytest.approx(1 / 3)
    assert mrr(RANKED, ["z"]) == 0.0


def test_ndcg_hand_computed():
    # gold = {b, d}: hits at ranks 2 and 4 -> DCG = 1/log2(3) + 1/log2(5); IDCG = 1 + 1/log2(3)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, ["b", "d"], k=5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(RANKED, ["a"], k=5) == 1.0


def test_first_relevant_rank():
    assert first_relevant_rank(RANKED, ["d"]) == 4
    assert first_relevant_rank(RANKED, ["z"]) is None


def test_corpus_normalized_variants():
    r = ranking_metrics(RANKED, ["a", "zz"], k=5, corpus_pmids={"a", "b", "c", "d", "e"})
    assert r.recall_at_k == 0.5 and r.recall_at_k_norm == 1.0
    assert r.n_gold == 2 and r.n_gold_in_corpus == 1
    none = ranking_metrics(RANKED, ["zz"], k=5, corpus_pmids={"a"})
    assert none.recall_at_k_norm is None and none.recall_at_k == 0.0


def test_empty_gold_rejected():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, [], k=5)
