"""Load profiles from configs/models.yaml, apply environment overrides, build adapters.

Resolution order for a role:
  1. ``LITBIORAG_MODEL__<ROLE>`` environment variable (spec string, see ModelSpec.parse)
  2. the selected profile's entry (profile from ``LITBIORAG_MODEL_PROFILE`` or the YAML default)

Validation enforces two production rules:
  * independence — generator, verifier and offline judge must come from different model
    families (the thesis reused one NLI family across selection, gating and evaluation);
  * calibration — thresholds are only returned for the model they were calibrated on.
Both rules are relaxed for test doubles so CI can run without weights.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from models.registry.schema import ModelSpec, Profile

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
ENV_PROFILE = "LITBIORAG_MODEL_PROFILE"
ENV_PREFIX = "LITBIORAG_MODEL__"

_ADAPTERS: dict[tuple[str, str], Callable[[ModelSpec], Any]] = {}


class RegistryError(RuntimeError):
    pass


def adapter(kind: str, role: str) -> Callable[[Callable[[ModelSpec], Any]], Callable]:
    """Register a factory for (kind, role). Factories receive the ModelSpec."""

    def deco(fn: Callable[[ModelSpec], Any]) -> Callable[[ModelSpec], Any]:
        _ADAPTERS[(kind, role)] = fn
        return fn

    return deco


def registered_adapters() -> list[tuple[str, str]]:
    return sorted(_ADAPTERS)


class ModelRegistry:
    def __init__(self, profile: Profile, source: str = ""):
        self.profile = profile
        self.source = source
        self._cache: dict[str, Any] = {}

    # ---- loading ----------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        path: Path | str = DEFAULT_PATH,
        profile: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ModelRegistry:
        env = os.environ if env is None else env
        raw = yaml.safe_load(Path(path).read_text()) or {}
        profiles = raw.get("profiles") or {}
        name = profile or env.get(ENV_PROFILE) or raw.get("default_profile")
        if name not in profiles:
            raise RegistryError(f"profile {name!r} not in {path} (have {sorted(profiles)})")
        body = dict(profiles[name])
        models: dict[str, ModelSpec] = {}
        for role, entry in (body.get("models") or {}).items():
            if isinstance(entry, str):
                models[role] = ModelSpec.parse(entry, role)
            else:
                models[role] = ModelSpec(role=role, **entry)
        for key, value in env.items():
            if key.startswith(ENV_PREFIX) and value:
                role = key[len(ENV_PREFIX) :].lower()
                models[role] = ModelSpec.parse(value, role)
        prof = Profile(
            name=name,
            description=body.get("description", ""),
            models=models,
            require_calibrated=bool(body.get("require_calibrated", True)),
            independent_families=body.get(
                "independent_families", [["generator", "verifier", "offline_judge"]]
            ),
        )
        reg = cls(prof, source=str(path))
        reg.validate()
        return reg

    # ---- validation -------------------------------------------------------------------
    def validate(self) -> None:
        # asymmetric retrievers: the article tower must come with its query tower
        from app.retrieval.encoders import PAIRED_QUERY_ENCODERS

        enc = self.profile.models.get("dense_encoder")
        if enc is not None and enc.model_id in PAIRED_QUERY_ENCODERS:
            want = PAIRED_QUERY_ENCODERS[enc.model_id]
            got = enc.params.get("query") or enc.params.get("query_model_id")
            if got and got != want:
                raise RegistryError(
                    f"profile {self.profile.name}: dense_encoder {enc.model_id!r} is paired with "
                    f"{want!r}, not {got!r}"
                )
        for group in self.profile.independent_families:
            seen: dict[str, str] = {}
            for role in group:
                spec = self.profile.models.get(role)
                if spec is None or spec.is_test_double:
                    continue
                if spec.family in seen:
                    raise RegistryError(
                        f"profile {self.profile.name}: roles {seen[spec.family]!r} and {role!r} "
                        f"share model family {spec.family!r}; verification must be independent "
                        "of generation (PRODUCTION_AUDIT.md §8)"
                    )
                seen[spec.family] = role

    # ---- access -----------------------------------------------------------------------
    def roles(self) -> list[str]:
        return sorted(self.profile.models)

    def spec(self, role: str) -> ModelSpec:
        try:
            return self.profile.models[role]
        except KeyError as e:
            raise RegistryError(
                f"profile {self.profile.name} has no model for role {role!r}"
            ) from e

    def thresholds(self, role: str) -> dict[str, float]:
        spec = self.spec(role)
        t = spec.thresholds()
        if t is None:
            if self.profile.require_calibrated and not spec.is_test_double:
                raise RegistryError(
                    f"{role} = {spec.name} has no calibration for this model id; run the "
                    "calibration job and record it under models.yaml before serving"
                )
            return {}
        return t

    def build(self, role: str, *, cache: bool = True) -> Any:
        if cache and role in self._cache:
            return self._cache[role]
        spec = self.spec(role)
        obj = wrap_with_cache(role, build_spec(spec))
        if cache:
            self._cache[role] = obj
        return obj

    def describe(self) -> dict[str, dict]:
        return {
            role: {
                "name": s.name,
                "family": s.family,
                "quantization": s.quantization,
                "endpoint": s.endpoint,
                "calibrated": s.thresholds() is not None,
            }
            for role, s in sorted(self.profile.models.items())
        }


def build_spec(spec: ModelSpec) -> Any:
    import models.registry.adapters  # noqa: F401  (registers factories)

    try:
        factory = _ADAPTERS[(spec.kind, spec.role)]
    except KeyError as e:
        raise RegistryError(
            f"no adapter registered for kind={spec.kind!r} role={spec.role!r}; "
            f"registered: {registered_adapters()}"
        ) from e
    return factory(spec)


def wrap_with_cache(role: str, obj: Any) -> Any:
    """Wrap verifier / reranker in the cross-request micro-batcher and verifier / dense_encoder in
    the on-disk cache when configs/resources.yaml enables them (cache outside, batcher inside, so
    cache hits never wait). Cache keys include the model name, so a swapped model never reads
    another model's results."""
    try:
        from app.core.budget import ResourceConfig

        res = ResourceConfig.from_yaml()
    except Exception:  # noqa: BLE001  (no config → no wrappers)
        return obj
    from app.core.batcher import wrap_with_batching

    obj = wrap_with_batching(role, obj, res.batching)
    cfg = res.cache or {}
    if not cfg.get("enabled"):
        return obj
    from app.core.cache import CachedEncoder, CachedVerifier, DiskCache

    store = DiskCache(cfg.get("path", "data/cache/litbiorag.sqlite"),
                      ttl_days=float(cfg.get("ttl_days", 30)))  # fmt: skip
    if role == "verifier" and cfg.get("verifier_pairs", True):
        return CachedVerifier(obj, store)
    if role == "dense_encoder" and cfg.get("query_embeddings", True):
        return CachedEncoder(obj, store)
    return obj
