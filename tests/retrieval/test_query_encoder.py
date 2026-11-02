"""Asymmetric encoders: searches use the query tower; MedCPT pairing is enforced by the registry."""

import pytest

from app.core.cache import CachedEncoder, DiskCache
from app.retrieval.dense import DenseIndex
from app.retrieval.encoders import PAIRED_QUERY_ENCODERS, HashEncoder, encode_queries
from models.registry import ModelRegistry, ModelSpec
from models.registry.loader import RegistryError


def test_search_uses_the_query_tower(fixture_bundle):
    enc = HashEncoder(128)
    idx = DenseIndex(fixture_bundle / "dense", enc)
    assert enc.query_calls == 0
    idx.search("MAP3K3 NF-kappaB", k=3)
    assert enc.query_calls == 1
    idx.search_many(["a", "b"], k=2)
    assert enc.query_calls == 3


def test_cached_encoder_separates_query_and_passage_spaces(tmp_path):
    enc = CachedEncoder(HashEncoder(32), DiskCache(tmp_path / "c.sqlite"))
    encode_queries(enc, ["x"])
    enc.encode(["x"])
    assert enc.cache.misses == 2  # different namespaces: a query vector never serves as a passage


def test_models_yaml_pairs_medcpt_everywhere():
    for profile in ("thesis-baseline", "hosted", "default"):
        spec = ModelRegistry.load(profile=profile).spec("dense_encoder")
        if spec.model_id in PAIRED_QUERY_ENCODERS:
            assert spec.params.get("query") == PAIRED_QUERY_ENCODERS[spec.model_id], profile


def test_registry_refuses_wrong_query_tower(monkeypatch):
    reg = ModelRegistry.load(profile="default")
    bad = ModelSpec.parse("sentence_transformers:ncbi/MedCPT-Article-Encoder?query=other/model",
                          "dense_encoder")  # fmt: skip
    reg.profile.models["dense_encoder"] = bad
    with pytest.raises(RegistryError):
        reg.validate()
