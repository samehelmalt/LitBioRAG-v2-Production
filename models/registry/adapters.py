"""Adapter factories: (kind, role) -> constructed object.

Adding a new *kind* is one function here. Swapping a *model* of an existing kind touches no code.
Generator, verifier and entity-extractor adapters are registered by their own packages
(app/generation, app/verification) when those land; the registry reports a clear error for a kind
that has no adapter yet instead of guessing.
"""

from __future__ import annotations

from models.registry.loader import adapter
from models.registry.schema import ModelSpec


@adapter("hash", "dense_encoder")
def _hash_encoder(spec: ModelSpec):
    from app.retrieval.encoders import HashEncoder

    return HashEncoder(int(spec.model_id))


@adapter("sentence_transformers", "dense_encoder")
def _st_encoder(spec: ModelSpec):
    from app.retrieval.encoders import SentenceTransformerEncoder

    return SentenceTransformerEncoder(
        spec.model_id,
        device=spec.device,
        max_seq_length=spec.max_tokens,
        revision=spec.revision,
    )


@adapter("lexical", "reranker")
def _lexical_reranker(spec: ModelSpec):
    from app.reranking.base import LexicalOverlapReranker

    return LexicalOverlapReranker()


@adapter("hf_seq_cls", "reranker")
def _ce_reranker(spec: ModelSpec):
    from app.reranking.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker(
        spec.model_id,
        device=spec.device,
        batch_size=int(spec.params.get("batch_size", 32)),
        max_length=spec.max_tokens,
        dtype=str(spec.params.get("dtype", "float16")),
        revision=spec.revision,
    )
