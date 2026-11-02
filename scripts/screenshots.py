"""Regenerate the landing-page screenshots from a running server.

    python -m scripts.screenshots --base-url http://127.0.0.1:8471 [--out app/web/static/shots]

Needs the JWT secret in LITBIORAG_JWT_SECRET (to mint a reviewer token for the review page) and
Node Playwright with Chromium (``node -e "require('playwright')"``). Called by scripts/release.py
when Playwright is present; otherwise the committed PNGs are kept.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def playwright_available() -> bool:
    node = shutil.which("node")
    if not node:
        return False
    try:
        return subprocess.run([node, "-e", "require('playwright')"], capture_output=True,
                              timeout=30, cwd=ROOT).returncode == 0  # fmt: skip
    except Exception:  # noqa: BLE001
        return False


def reviewer_token(secret: str) -> str:
    import jwt

    return jwt.encode({"sub": "shots-reviewer", "roles": ["reviewer"], "courses": [],
                       "aud": "litbiorag", "exp": int(time.time()) + 3600}, secret,
                      algorithm="HS256")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8471")
    ap.add_argument("--out", default="app/web/static/shots")
    ap.add_argument("--theme", default="dark", choices=["dark", "light"])
    a = ap.parse_args(argv)
    secret = os.environ.get("LITBIORAG_JWT_SECRET", "")
    if len(secret) < 32:
        print("LITBIORAG_JWT_SECRET (32+ chars) is required", file=sys.stderr)
        return 2
    if not playwright_available():
        print("Node Playwright not available; keeping committed screenshots", file=sys.stderr)
        return 3
    Path(a.out).mkdir(parents=True, exist_ok=True)
    cmd = ["node", str(ROOT / "scripts" / "screenshots.js"), a.base_url, reviewer_token(secret),
           a.out, a.theme]  # fmt: skip
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
