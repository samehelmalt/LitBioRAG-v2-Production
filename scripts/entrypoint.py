"""Container entrypoint: make the service runnable with zero preparation, then serve.

1. If LITBIORAG_BUNDLE does not exist, build a small demo bundle from scripts/demo/demo_docs.jsonl
   with the hash encoder (profile `test`): the UI, API, uploads, review queue and corpus pages
   all work; answers come from deterministic test doubles, not a real model.
2. If LITBIORAG_JWT_SECRET is unset, generate one for this container run (printed once) so the
   demo tokens below can be issued; set it in .env for anything persistent.
3. Print ready-to-paste demo tokens (user, reviewer, admin) and start uvicorn.

Real deployments set LITBIORAG_BUNDLE to a fetched bundle and LITBIORAG_MODEL_PROFILE=hosted (or a
GPU profile) and never rely on the generated secret.
"""

from __future__ import annotations

import os
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    bundle = Path(os.environ.get("LITBIORAG_BUNDLE") or ROOT / "data" / "artifacts" / "demo")
    profile = os.environ.get("LITBIORAG_MODEL_PROFILE") or ""
    if not (bundle / "manifest.json").exists():
        from scripts.build_indexes import main as build

        print(f"[entrypoint] no bundle at {bundle}; building the demo bundle", flush=True)
        rc = build(["--docs", str(ROOT / "scripts" / "demo" / "demo_docs.jsonl"), "--out",
                    str(bundle), "--corpus-id", "demo", "--encoder", "hash:128",
                    "--shard-size", "25"])  # fmt: skip
        if rc != 0:
            print("[entrypoint] demo bundle build failed", file=sys.stderr)
            return rc
        if not profile:
            profile = "test"
    os.environ["LITBIORAG_BUNDLE"] = str(bundle)
    os.environ["LITBIORAG_MODEL_PROFILE"] = (profile
                                             or os.environ.get("LITBIORAG_MODEL_PROFILE", "test"))
    if not os.environ.get("LITBIORAG_JWT_SECRET"):
        os.environ["LITBIORAG_JWT_SECRET"] = secrets.token_urlsafe(48)
        print("[entrypoint] LITBIORAG_JWT_SECRET was unset: generated one for this run "
              "(tokens below stop working on restart; set it in .env)", flush=True)
    try:
        import jwt

        s = os.environ["LITBIORAG_JWT_SECRET"]
        exp = int(time.time()) + 30 * 86400

        def tok(sub, roles):
            return jwt.encode({"sub": sub, "roles": roles, "courses": [], "aud": "litbiorag",
                               "exp": exp}, s, algorithm="HS256")  # fmt: skip

        print("[entrypoint] demo tokens (paste into the UI 'Sign in'):", flush=True)
        for sub, roles in (("demo-user", []), ("demo-reviewer", ["reviewer"]),
                           ("demo-admin", ["admin", "reviewer"])):
            print(f"  {sub:14s} {tok(sub, roles)}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[entrypoint] could not issue demo tokens: {e}", flush=True)
    from scripts.serve import main as serve

    host = os.environ.get("LITBIORAG_API_HOST", "0.0.0.0")
    port = os.environ.get("LITBIORAG_API_PORT", "8471")
    print(f"[entrypoint] profile={os.environ['LITBIORAG_MODEL_PROFILE']} bundle={bundle} "
          f"→ http://localhost:{port}", flush=True)
    return serve(["--host", host, "--port", port])


if __name__ == "__main__":
    raise SystemExit(main())
