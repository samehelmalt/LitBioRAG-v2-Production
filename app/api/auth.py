"""Authentication (who) and entitlement (which courses) — kept separate on purpose.

* JWT bearer tokens (HS256 by default; secret from the environment variable named in api.yaml).
  Claims used: ``sub``, ``courses`` (list of course ids), ``roles`` (list), ``aud``.
* Static API keys for server-to-server use: ``LITBIORAG_API_KEYS="key1:service-a,key2:service-b"``.
  An API key grants no course entitlements.
* Anonymous access is allowed only when ``auth.required`` is false, and then only to the public
  corpus plus ``auth.anonymous_courses``.
Entitlement is enforced **before retrieval** (``course_ids`` filter); a request for a course the
principal does not hold is refused with 403, never silently narrowed.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Callable
from dataclasses import dataclass, field

from app.api.settings import AuthSettings


class AuthError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class Principal:
    sub: str
    courses: frozenset[str] = field(default_factory=frozenset)
    roles: frozenset[str] = field(default_factory=frozenset)
    anonymous: bool = False
    via: str = "anonymous"  # jwt | api_key | anonymous

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles


def _api_keys(cfg: AuthSettings) -> dict[str, str]:
    raw = os.environ.get(cfg.api_keys_env, "")
    out: dict[str, str] = {}
    for item in filter(None, (s.strip() for s in raw.split(","))):
        key, _, subject = item.partition(":")
        out[key] = subject or "api-key"
    return out


def authenticate(
    cfg: AuthSettings, *, authorization: str | None, api_key: str | None,
    key_lookup: Callable[[str], str | None] | None = None,
) -> Principal:
    if api_key:
        for known, subject in _api_keys(cfg).items():
            if hmac.compare_digest(known, api_key):
                return Principal(sub=subject, via="api_key")
        owner = key_lookup(api_key) if key_lookup else None
        if owner:
            return Principal(sub=owner, via="api_key")
        raise AuthError(401, "invalid API key")
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise AuthError(401, "malformed Authorization header")
        secret = os.environ.get(cfg.jwt_secret_env, "")
        if not secret:
            raise AuthError(503, "token authentication is not configured on this server")
        try:
            import jwt  # PyJWT

            claims = jwt.decode(
                token, secret, algorithms=cfg.jwt_algorithms, audience=cfg.jwt_audience
            )
        except Exception as e:  # noqa: BLE001  (every decode failure is a 401; detail is generic)
            raise AuthError(401, "invalid or expired token") from e
        return Principal(
            sub=str(claims.get("sub", "")) or "unknown",
            courses=frozenset(map(str, claims.get("courses", []) or [])),
            roles=frozenset(map(str, claims.get("roles", []) or [])),
            via="jwt",
        )
    if cfg.required:
        raise AuthError(401, "authentication required")
    return Principal(sub="anonymous", courses=frozenset(cfg.anonymous_courses), anonymous=True)


def check_entitlement(principal: Principal, course_ids: list[str] | None) -> list[str] | None:
    """Return the course filter to apply, or raise 403. Admins may query any course."""
    if not course_ids:
        return None
    if principal.is_admin:
        return list(course_ids)
    missing = sorted(set(course_ids) - principal.courses)
    if missing:
        raise AuthError(403, f"not entitled to course(s): {', '.join(missing)}")
    return list(course_ids)
