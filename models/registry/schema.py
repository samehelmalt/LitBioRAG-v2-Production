"""Registry schema.

A ``ModelSpec`` says *which* model fills a role and *how* it is loaded; an adapter keyed by
``kind`` knows how to build it. Swapping a model is therefore a YAML or environment change:

    LITBIORAG_MODEL__GENERATOR="openai_compatible:Qwen/Qwen3-14B@main?endpoint=http://vllm:8000/v1"

Thresholds are attached to the model they were calibrated for (``Calibration.for_model``). When
the model changes, the thresholds become invalid until a new calibration is recorded; the
registry refuses to hand out stale thresholds outside test profiles.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl

from pydantic import BaseModel, Field, field_validator, model_validator

ROLES = (
    "dense_encoder",
    "reranker",
    "generator",
    "verifier",
    "entity_extractor",
    "offline_judge",  # evaluation only; never used at runtime
)

KINDS = {
    "dense_encoder": {"sentence_transformers", "hash"},
    "reranker": {"hf_seq_cls", "lexical"},
    "generator": {"hf_causal", "openai_compatible", "anthropic", "template"},
    "verifier": {"hf_seq_cls", "minicheck", "overlap"},
    "entity_extractor": {"regex", "scispacy"},
    "offline_judge": {"openai_compatible", "anthropic", "hf_causal", "template"},
}

TEST_KINDS = {"hash", "lexical", "template", "overlap", "regex"}

_SPEC_RE = re.compile(r"^(?P<kind>[a-z_]+):(?P<model>[^@?]+)(?:@(?P<rev>[^?]+))?(?:\?(?P<q>.*))?$")


def model_family(model_id: str) -> str:
    """Coarse family used for the independence rule (generator ≠ verifier ≠ judge).

    Hugging Face ids give the organisation; well-known model lines are normalised so that
    e.g. 'Qwen/Qwen3-8B' and 'Qwen/Qwen3-Reranker-0.6B' are both 'qwen'.
    """
    m = model_id.strip().lower()
    for key, fam in (
        ("qwen", "qwen"), ("llama", "llama"), ("gemma", "gemma"), ("mistral", "mistral"),
        ("deepseek", "deepseek"), ("phi", "phi"), ("claude", "anthropic"), ("gpt", "openai"),
        ("deberta", "deberta"), ("roberta", "roberta"), ("bert", "bert"),
        ("minicheck", "minicheck"), ("medcpt", "medcpt"), ("bge", "bge"), ("e5", "e5"),
        ("t5", "t5"),
    ):  # fmt: skip
        if key in m:
            return fam
    return m.split("/")[0] if "/" in m else m


class Calibration(BaseModel):
    """Thresholds and where they came from. Valid only for ``for_model``."""

    for_model: str
    thresholds: dict[str, float] = Field(default_factory=dict)
    validated_on: str = ""  # dataset / split id
    date: str = ""
    notes: str = ""


class ModelSpec(BaseModel):
    role: str
    kind: str
    model_id: str
    revision: str | None = None
    quantization: str | None = Field(default=None, description="none | 8bit | 4bit")
    device: str | None = None
    max_tokens: int | None = None
    endpoint: str | None = None
    api_key_env: str | None = None
    family: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    calibration: Calibration | None = None

    @field_validator("role")
    @classmethod
    def _role(cls, v: str) -> str:
        if v not in ROLES:
            raise ValueError(f"unknown role {v!r}; roles: {ROLES}")
        return v

    @field_validator("quantization")
    @classmethod
    def _quant(cls, v: str | None) -> str | None:
        if v in (None, "", "none"):
            return None
        if v not in ("8bit", "4bit"):
            raise ValueError("quantization must be none, 8bit or 4bit")
        return v

    @model_validator(mode="after")
    def _kind_for_role(self) -> ModelSpec:
        if self.kind not in KINDS[self.role]:
            raise ValueError(
                f"kind {self.kind!r} is not valid for role {self.role!r}; "
                f"allowed: {sorted(KINDS[self.role])}"
            )
        if self.family is None:
            self.family = model_family(self.model_id)
        return self

    @property
    def is_test_double(self) -> bool:
        return self.kind in TEST_KINDS

    @property
    def name(self) -> str:
        rev = f"@{self.revision}" if self.revision else ""
        return f"{self.kind}:{self.model_id}{rev}"

    def thresholds(self) -> dict[str, float] | None:
        """Calibrated thresholds, or None when none exist or they belong to another model."""
        if self.calibration is None or self.calibration.for_model != self.model_id:
            return None
        return dict(self.calibration.thresholds)

    @classmethod
    def parse(cls, spec: str, role: str) -> ModelSpec:
        """'kind:model_id[@revision][?key=value&...]' → ModelSpec."""
        m = _SPEC_RE.match(spec.strip())
        if not m:
            raise ValueError(
                f"bad model spec {spec!r}; expected kind:model_id[@revision][?k=v&k=v]"
            )
        extras = dict(parse_qsl(m.group("q") or "", keep_blank_values=False))
        known = {
            k: extras.pop(k)
            for k in ("quantization", "device", "endpoint", "api_key_env", "family")
            if k in extras
        }
        if "max_tokens" in extras:
            known["max_tokens"] = int(extras.pop("max_tokens"))
        return cls(
            role=role,
            kind=m.group("kind"),
            model_id=m.group("model"),
            revision=m.group("rev"),
            params=extras,
            **known,
        )


class Profile(BaseModel):
    name: str
    description: str = ""
    models: dict[str, ModelSpec]
    require_calibrated: bool = Field(
        default=True,
        description="refuse thresholds() for a model without calibration for its own id",
    )
    independent_families: list[list[str]] = Field(
        default_factory=lambda: [["generator", "verifier", "offline_judge"]],
        description="role groups that must not share a model family",
    )

    @model_validator(mode="after")
    def _roles_match(self) -> Profile:
        for role, spec in self.models.items():
            if spec.role != role:
                raise ValueError(f"profile {self.name}: entry {role} has role {spec.role}")
        return self
