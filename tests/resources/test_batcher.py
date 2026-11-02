"""Micro-batching never changes results: batched == direct, order kept, errors propagate to every
caller, a lone caller takes the direct path, and the registry wraps the configured roles."""

import threading
import time

import pytest

from app.core.batcher import BatchedReranker, BatchedVerifier, MicroBatcher, wrap_with_batching
from app.reranking.base import Candidate, LexicalOverlapReranker
from app.verification.base import OverlapVerifier


class SlowSquare:
    """Item-wise, order-preserving fn that records how many calls and the batch sizes."""

    def __init__(self):
        self.calls: list[int] = []
        self.lock = threading.Lock()

    def __call__(self, items):
        with self.lock:
            self.calls.append(len(items))
        time.sleep(0.03)
        return [x * x for x in items]


def test_lone_caller_goes_direct():
    fn = SlowSquare()
    b = MicroBatcher(fn, max_wait_ms=50)
    assert b.submit([1, 2, 3]) == [1, 4, 9]
    assert b.submit([]) == []
    assert fn.calls == [3] and b.stats()["direct_calls"] == 1 and b.stats()["batches"] == 0


def test_concurrent_callers_share_a_pass_and_get_their_own_slices():
    fn = SlowSquare()
    b = MicroBatcher(fn, max_wait_ms=40, max_batch=1000)
    results: dict[int, list] = {}

    def worker(k):
        results[k] = b.submit(list(range(k * 10, k * 10 + k + 1)))

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(1, 7)]
    for t in ts:
        t.start()
        time.sleep(0.004)  # arrive while the first call is still running
    for t in ts:
        t.join(5)
    for k, out in results.items():
        assert out == [x * x for x in range(k * 10, k * 10 + k + 1)]
    s = b.stats()
    assert s["batches"] >= 1 and s["batched_items"] + sum(fn.calls[:1]) <= sum(fn.calls)
    assert len(fn.calls) < 6  # fewer forward passes than callers
    assert s["max_batch"] >= 2


def test_max_batch_closes_the_window_early_and_errors_reach_everyone():
    def boom(items):
        time.sleep(0.05)  # long enough for the next callers to arrive while this one runs
        if any(x < 0 for x in items):
            raise ValueError("negative")
        return items

    b = MicroBatcher(boom, max_wait_ms=500, max_batch=4)
    errors: list[str] = []
    oks: list[list] = []

    def w(items):
        try:
            oks.append(b.submit(items))
        except ValueError as e:
            errors.append(str(e))

    hold = threading.Thread(target=w, args=([100, 101],))  # direct path; keeps _inside > 0
    t0 = time.monotonic()
    ts = [hold, threading.Thread(target=w, args=([-1, 2],)),
          threading.Thread(target=w, args=([3, 4],))]  # fmt: skip
    for t in ts:
        t.start()
        time.sleep(0.003)
    for t in ts:
        t.join(5)
    assert time.monotonic() - t0 < 0.45  # 4 items reached max_batch before the 500 ms window
    assert errors == ["negative", "negative"] and oks == [[100, 101]]


def _concurrent(fn, n_threads: int, make_args):
    out: dict[int, object] = {}

    def w(i):
        out[i] = fn(*make_args(i))

    ts = [threading.Thread(target=w, args=(i,)) for i in range(n_threads)]
    for t in ts:
        t.start()
        time.sleep(0.002)
    for t in ts:
        t.join(5)
    return [out[i] for i in range(n_threads)]


def test_batched_verifier_matches_direct_scores():
    inner = OverlapVerifier()
    bv = BatchedVerifier(inner, max_wait_ms=30)
    pairs_by_req = [
        [("Atorvastatin lowers LDL cholesterol.", "Atorvastatin lowers LDL."),
         ("MAP3K7 activates NF-kB signalling.", "MAP3K7 does not activate NF-kB.")],
        [("Potassium above 6 mmol/L is dangerous.", "Hyperkalaemia above 6 is dangerous.")],
        [("The study found no effect.", "The study found an effect."),
         ("Sodium is low.", "Sodium is low."), ("x y z", "unrelated words here")],
    ]
    batched = _concurrent(bv.score_pairs, len(pairs_by_req), lambda i: (pairs_by_req[i],))
    for req, got in zip(pairs_by_req, batched, strict=True):
        assert got == inner.score_pairs(req)
    assert bv.name == inner.name


class PairScoringLexical:
    """Reranker double exposing score_pairs, so it can be batched (CrossEncoderReranker shape)."""

    name = "lexical-pairs"

    def score_pairs(self, pairs):
        return [float(len(set(q.lower().split()) & set(t.lower().split()))) for q, t in pairs]

    def rerank(self, query, candidates, top_n):
        from app.reranking.base import RerankHit

        s = self.score_pairs([(query, c.text) for c in candidates])
        hits = [RerankHit(c.passage_id, x) for c, x in zip(candidates, s, strict=True)]
        hits.sort(key=lambda h: (-h.score, h.passage_id))
        return hits[:top_n]


def test_batched_reranker_matches_direct_and_lexical_is_left_alone():
    inner = PairScoringLexical()
    br = BatchedReranker(inner, max_wait_ms=30)
    cands = [Candidate("p1", "potassium channel in the heart"), Candidate("p2", "sodium pump"),
             Candidate("p3", "heart potassium levels")]
    queries = ["potassium heart", "sodium", "heart"]
    got = _concurrent(br.rerank, 3, lambda i: (queries[i], cands, 2))
    for q, hits in zip(queries, got, strict=True):
        assert hits == inner.rerank(q, cands, 2)
    with pytest.raises(TypeError):
        BatchedReranker(LexicalOverlapReranker())
    cfg = {"enabled": True, "roles": ["verifier", "reranker"]}
    lex = LexicalOverlapReranker()
    assert wrap_with_batching("reranker", lex, cfg) is lex  # no score_pairs → unchanged
    assert isinstance(wrap_with_batching("reranker", inner, cfg), BatchedReranker)
    assert isinstance(wrap_with_batching("verifier", OverlapVerifier(), cfg), BatchedVerifier)
    v = OverlapVerifier()
    assert wrap_with_batching("verifier", v, {"enabled": False}) is v
    assert wrap_with_batching("generator", v, cfg) is v


def test_registry_wraps_verifier_when_enabled():
    from models.registry.loader import ModelRegistry

    reg = ModelRegistry.load(profile="test")
    v = reg.build("verifier", cache=False)
    assert isinstance(v, BatchedVerifier)  # resources.yaml: batching.enabled is true by default
    assert v.score_pairs([("potassium lowers heart rate", "potassium lowers heart rate")])[0] \
        .entail > 0.9
