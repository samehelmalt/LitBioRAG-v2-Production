from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]


class AuthSettings(BaseModel):
    required: bool = False
    jwt_secret_env: str = "LITBIORAG_JWT_SECRET"
    jwt_algorithms: list[str] = Field(default_factory=lambda: ["HS256"])
    jwt_audience: str = "litbiorag"
    api_keys_env: str = "LITBIORAG_API_KEYS"
    anonymous_courses: list[str] = Field(default_factory=list)


class LimitSettings(BaseModel):
    max_question_chars: int = 1000
    max_units: int = 12
    rate_per_minute: int = 30
    burst: int = 10


class AuditSettings(BaseModel):
    path: str = "data/audit/requests.jsonl"
    store_question_text: bool = True


class ApiSettings(BaseModel):
    bundle: str = ""
    profile: str = ""
    scope_profile: str = "research"
    auth: AuthSettings = Field(default_factory=AuthSettings)
    limits: LimitSettings = Field(default_factory=LimitSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    cors_origins: list[str] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str | None = None, **overrides: Any) -> ApiSettings:
        p = Path(path or os.environ.get("LITBIORAG_API_CONFIG") or ROOT / "configs" / "api.yaml")
        raw = yaml.safe_load(p.read_text()) if p.exists() else {}
        raw = {**(raw or {}), **{k: v for k, v in overrides.items() if v is not None}}
        s = cls.model_validate(raw)
        s.bundle = os.environ.get("LITBIORAG_BUNDLE", "") or s.bundle
        s.profile = os.environ.get("LITBIORAG_MODEL_PROFILE", "") or s.profile
        return s

    @property
    def bundle_path(self) -> Path:
        p = Path(self.bundle)
        return p if p.is_absolute() else ROOT / p

    @property
    def audit_path(self) -> Path | None:
        if not self.audit.path:
            return None
        p = Path(self.audit.path)
        return p if p.is_absolute() else ROOT / p
