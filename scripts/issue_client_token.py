"""Issue a client (prospect) token on the host from the server's JWT secret.

    python -m scripts.issue_client_token --subject prospect-001 --days 7 [--courses IM-2026]
                                         [--tenant clinic-a] [--roles reviewer]

Reads LITBIORAG_JWT_SECRET from the environment or from ./.env (KEY=VALUE lines; no extra
dependency). Refuses demo subjects, short secrets and validity over 30 days. The token is printed
once; it is never stored by this tool.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_SUBJECTS = {"anonymous", "demo-user", "demo-reviewer", "demo-admin"}


def read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.split("#", 1)[0].strip().strip('"').strip("'")
    return out


def issue(secret: str, *, subject: str, days: int, courses: list[str], roles: list[str],
          tenant: str = "", audience: str = "litbiorag") -> str:  # fmt: skip
    import jwt

    if subject in DEMO_SUBJECTS or not subject.strip():
        raise ValueError("choose a unique prospect identifier")
    if not 1 <= days <= 30:
        raise ValueError("days must be between 1 and 30")
    if len(secret) < 32:
        raise ValueError("LITBIORAG_JWT_SECRET must be at least 32 characters")
    claims = {"sub": subject, "roles": roles, "courses": courses, "aud": audience,
              "iat": int(time.time()), "exp": int(time.time()) + days * 86400}  # fmt: skip
    if tenant:
        claims["tenant"] = tenant
    return jwt.encode(claims, secret, algorithm="HS256")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--courses", default="")
    ap.add_argument("--roles", default="")
    ap.add_argument("--tenant", default="")
    a = ap.parse_args(argv)
    secret = os.environ.get("LITBIORAG_JWT_SECRET") or read_env_file(ROOT / ".env").get(
        "LITBIORAG_JWT_SECRET", "")
    try:
        tok = issue(secret, subject=a.subject, days=a.days,
                    courses=[c.strip() for c in a.courses.split(",") if c.strip()],
                    roles=[r.strip() for r in a.roles.split(",") if r.strip()],
                    tenant=a.tenant)  # fmt: skip
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(tok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
