"""Adapter factories: (kind, role) -> constructed object.

Adding a new *kind* is one function here. Swapping a *model* of an existing kind touches no code.
Generator, verifier and entity-extractor adapters are registered by their own packages
(app/generation, app/verification) when those land; the registry reports a clear error for a kind
that has no adapter yet instead of guessing.
"""

from __future__ import annotations

import os

from models.registry.loader import adapter
from models.registry.schema import ModelSpec


@adapter("hash", "dense_encoder")
def _hash_encoder(spec: ModelSpec):
    from app.retrieval.encoders import HashEncoder

    return HashEncoder(int(spec.model_id))


@adapter("sentence_transformers", "dense_encoder")
def _st_encoder(spec: ModelSpec):
    from app.retrieval.encoders import PAIRED_QUERY_ENCODERS, SentenceTransformerEncoder

    query_id = spec.params.get("query") or spec.params.get("query_model_id") \
        or PAIRED_QUERY_ENCODERS.get(spec.model_id)
    return SentenceTransformerEncoder(
        spec.model_id,
        device=spec.device,
        max_seq_length=spec.max_tokens,
        revision=spec.revision,
        query_model_id=query_id,
        query_max_seq_length=int(spec.params["query_max_tokens"])
        if "query_max_tokens" in spec.params else None,
    )


@adapter("template", "generator")
def _template_generator(spec: ModelSpec):
    from app.generation.base import TemplateGenerator

    return TemplateGenerator(max_claims=int(spec.params.get("max_claims", 3)))


@adapter("hf_causal", "generator")
def _hf_generator(spec: ModelSpec):
    from app.generation.base import HFCausalGenerator

    return HFCausalGenerator(
        spec.model_id,
        revision=spec.revision,
        quantization=spec.quantization,
        device=spec.device,
        dtype=str(spec.params.get("dtype", "bfloat16")),
    )


@adapter("openai_compatible", "generator")
@adapter("openai_compatible", "offline_judge")
def _openai_generator(spec: ModelSpec):
    from app.generation.base import OpenAICompatibleGenerator

    endpoint = spec.endpoint or "http://localhost:8000/v1"
    if spec.endpoint and spec.endpoint.startswith("http://vllm:") and os.environ.get("VLLM_URL"):
        endpoint = os.environ["VLLM_URL"]  # vllm-8b: another host without editing yaml
    return OpenAICompatibleGenerator(
        spec.model_id,
        endpoint=endpoint,
        api_key_env=spec.api_key_env,
        json_mode=str(spec.params.get("json_mode", "true")).lower() == "true",
        timeout=int(spec.params.get("timeout_s", 60)),
        max_retries=int(spec.params.get("max_retries", 3)),
        backoff_s=float(spec.params.get("backoff_s", 2.0)),
    )


@adapter("anthropic", "generator")
@adapter("anthropic", "offline_judge")
def _anthropic_generator(spec: ModelSpec):
    from app.generation.base import AnthropicGenerator

    return AnthropicGenerator(spec.model_id, api_key_env=spec.api_key_env or "ANTHROPIC_API_KEY")


@adapter("overlap", "verifier")
def _overlap_verifier(spec: ModelSpec):
    from app.verification.base import OverlapVerifier

    return OverlapVerifier(min_overlap=float(spec.params.get("min_overlap", 0.6)))


@adapter("hf_seq_cls", "verifier")
def _nli_verifier(spec: ModelSpec):
    from app.verification.base import HFNLIVerifier

    return HFNLIVerifier(
        spec.model_id,
        revision=spec.revision,
        device=spec.device,
        batch_size=int(spec.params.get("batch_size", 16)),
        max_length=spec.max_tokens,
        dtype=str(spec.params.get("dtype", "float16")),
    )


@adapter("regex", "entity_extractor")
def _regex_entities(spec: ModelSpec):
    from app.verification.entities import RegexBiomedEntityExtractor

    return RegexBiomedEntityExtractor()


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
