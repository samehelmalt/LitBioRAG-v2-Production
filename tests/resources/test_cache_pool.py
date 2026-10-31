"""On-disk cache wrappers and GPU residency pool (no torch needed)."""

import pytest

from app.core.budget import ResourceConfig
from app.core.cache import CachedEncoder, CachedVerifier, DiskCache
from app.retrieval.encoders import HashEncoder
from app.verification.base import OverlapVerifier
from models.registry import ModelRegistry
from models.registry.pool import ModelPool, PreflightError


class CountingVerifier(OverlapVerifier):
    calls = 0

    def score_pairs(self, pairs):
        CountingVerifier.calls += len(pairs)
        return super().score_pairs(pairs)


def test_cached_verifier_scores_each_pair_once(tmp_path):
    cache = DiskCache(tmp_path / "c.sqlite")
    v = CachedVerifier(CountingVerifier(), cache)
    pairs = [("A reduces B.", "A reduces B."), ("A reduces B.", "A does not reduce B.")]
    first = v.score_pairs(pairs)
    second = v.score_pairs(pairs + [("C", "C")])
    assert CountingVerifier.calls == 3
    assert [s.model_dump() for s in first] == [s.model_dump() for s in second[:2]]
    assert cache.hits == 2 and cache.misses == 3


def test_cache_key_includes_model_name(tmp_path):
    cache = DiskCache(tmp_path / "c.sqlite")
    a = CachedVerifier(OverlapVerifier(), cache)
    b = CachedVerifier(OverlapVerifier(), cache)
    b.name = "other-model"
    a.score_pairs([("x y z", "x y z")])
    b.score_pairs([("x y z", "x y z")])
    assert cache.misses == 2  # a swapped model never reads the old model's scores


def test_cached_encoder_roundtrip(tmp_path):
    enc = CachedEncoder(HashEncoder(32), DiskCache(tmp_path / "c.sqlite"))
    v1 = enc.encode(["map3k3 signalling"])
    v2 = enc.encode(["map3k3 signalling", "other"])
    assert v1.shape == (1, 32) and v2.shape == (2, 32)
    assert (v1[0] == v2[0]).all() and enc.cache.hits == 1


def test_cache_ttl_expiry(tmp_path):
    cache = DiskCache(tmp_path / "c.sqlite", ttl_days=0.0000001)
    cache.put("ns", "k", {"a": 1})
    import time

    time.sleep(0.02)
    assert cache.get("ns", "k") is None


def _resources(**gpu) -> ResourceConfig:
    base = {"vram_gb": 12, "residency": {"generator": "on_demand"},
            "vram_estimate_gb": {"dense_encoder": 0.5, "reranker": 1.2, "verifier": 1.0,
                                 "generator": 6.0}}  # fmt: skip
    base.update(gpu)
    return ResourceConfig(gpu=base)


def test_pool_residency_and_preflight_with_test_doubles():
    reg = ModelRegistry.load(profile="test")
    pool = ModelPool(reg, _resources())
    rep = pool.preflight()
    assert rep["fits"] and rep["peak_gb"] == 0.0  # doubles and hosted models use no VRAM
    with pool.use("generator") as g1:
        pass
    with pool.use("generator") as g2:
        pass
    assert g1 is not g2 and pool.load_counts["generator"] == 2  # on_demand: rebuilt each time
    v1 = pool.acquire("verifier")
    v2 = pool.acquire("verifier")
    assert v1 is v2 and pool.load_counts["verifier"] == 1  # resident: built once


def test_preflight_refuses_profile_that_does_not_fit():
    reg = ModelRegistry.load(profile="default")  # local 8B generator + local models
    pool = ModelPool(reg, _resources(vram_gb=6))
    with pytest.raises(PreflightError):
        pool.preflight()
    ok = ModelPool(reg, _resources(vram_gb=12)).preflight()
    assert ok["fits"] and ok["peak_gb"] == pytest.approx(0.5 + 1.2 + 1.0 + 6.0)
    hosted = ModelPool(ModelRegistry.load(profile="hosted"), _resources(vram_gb=12)).preflight()
    assert hosted["on_demand_gb"]["generator"] == 0.0  # API generator costs no VRAM
