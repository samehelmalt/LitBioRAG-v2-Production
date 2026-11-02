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
    max_upload_mb: int = 50
    max_files_per_kb: int = 500


class KBSettings(BaseModel):
    enabled: bool = True
    root: str = "data/kb"
    default_scope: str = "both"  # indexed | uploaded | both


class AuditSettings(BaseModel):
    path: str = "data/audit/requests.jsonl"
    store_question_text: bool = True


class TrialSettings(BaseModel):
    questions: int = 0  # 0 = no trial limit; >0 = answered questions per authenticated identity
    db_path: str = "data/audit/trial_uses.sqlite"


class ApiSettings(BaseModel):
    bundle: str = ""
    profile: str = ""
    scope_profile: str = "research"
    trial: TrialSettings = Field(default_factory=TrialSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    limits: LimitSettings = Field(default_factory=LimitSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    kb: KBSettings = Field(default_factory=KBSettings)
    cors_origins: list[str] = Field(default_factory=list)
    state_backend: str = Field("sqlite", pattern="^(sqlite|redis)$",
                               description="sqlite = per process; redis = shared rate limiter, "
                                           "trial/usage ledgers and KB jobs across replicas")
    redis_url_env: str = "REDIS_URL"

    @property
    def redis_url(self) -> str:
        return os.environ.get(self.redis_url_env, "")

    @classmethod
    def load(cls, path: Path | str | None = None, **overrides: Any) -> ApiSettings:
        p = Path(path or os.environ.get("LITBIORAG_API_CONFIG") or ROOT / "configs" / "api.yaml")
        raw = yaml.safe_load(p.read_text()) if p.exists() else {}
        raw = {**(raw or {}), **{k: v for k, v in overrides.items() if v is not None}}
        s = cls.model_validate(raw)
        s.bundle = os.environ.get("LITBIORAG_BUNDLE", "") or s.bundle
        s.profile = os.environ.get("LITBIORAG_MODEL_PROFILE", "") or s.profile
        if os.environ.get("AEOS_TRIAL_QUESTIONS"):
            s.trial.questions = int(os.environ["AEOS_TRIAL_QUESTIONS"])
        if os.environ.get("AEOS_STATE_BACKEND"):
            s.state_backend = os.environ["AEOS_STATE_BACKEND"]
        return s

    @property
    def bundle_path(self) -> Path:
        p = Path(self.bundle)
        return p if p.is_absolute() else ROOT / p

    @property
    def kb_root(self) -> Path | None:
        if not self.kb.enabled:
            return None
        p = Path(self.kb.root)
        return p if p.is_absolute() else ROOT / p

    @property
    def audit_path(self) -> Path | None:
        if not self.audit.path:
            return None
        p = Path(self.audit.path)
        return p if p.is_absolute() else ROOT / p
