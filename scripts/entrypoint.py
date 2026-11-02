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


def _client_mode_guard(bundle: Path, profile: str) -> None:
    """AEOS_CLIENT_MODE=1 (client trials): refuse anything that could pass a demo off as the real
    product — the test doubles, the demo bundle, a per-run JWT secret."""
    problems = []
    if (profile or "test") == "test":
        problems.append("profile 'test' uses deterministic doubles; set a real profile")
    if bundle.name == "demo":
        problems.append("the demo bundle is not client content; set CORPUS_ID to a real bundle")
    if len(os.environ.get("LITBIORAG_JWT_SECRET", "")) < 32:
        problems.append("LITBIORAG_JWT_SECRET must be a persistent secret of 32+ characters")
    if problems:
        for p in problems:
            print(f"[entrypoint] client mode refused: {p}", file=sys.stderr, flush=True)
        raise SystemExit(3)


def _offline_demo(bundle: Path, profile: str) -> None:
    """Real small CPU models: prefetch (or verify) the weights and build a real-embedding bundle
    from the synthetic deck on first start."""
    from scripts.model_cache import ensure_models

    ensure_models(profile, offline=os.environ.get("AEOS_OFFLINE", "0") == "1")
    if not (bundle / "manifest.json").exists():
        from scripts.build_indexes import main as build
        from scripts.make_prototype_deck import main as make_deck

        out = ROOT / "data" / "prototype"
        make_deck(["--out", str(out)])
        from scripts.corpus.pptx_decks import extract_deck, write_jsonl

        docs = extract_deck(out / "potassium_disorders.pptx", course_id="IM-2026",
                            lecture_id="L03", lecture_title="Potassium disorders",
                            academic_year="2026", version="2026a")  # fmt: skip
        jsonl = out / "docs.jsonl"
        write_jsonl(docs, jsonl)
        from models.registry.loader import ModelRegistry

        enc = ModelRegistry.load(profile=profile).spec("dense_encoder")
        rc = build(["--docs", str(jsonl), "--out", str(bundle), "--corpus-id",
                    "synthetic-potassium-v1", "--encoder", f"sentence_transformers:{enc.model_id}",
                    "--shard-size", "25", "--batch-size", "8"])  # fmt: skip
        if rc != 0:
            raise SystemExit(f"[entrypoint] real-embedding demo bundle build failed: {rc}")


def main() -> int:
    bundle = Path(os.environ.get("LITBIORAG_BUNDLE") or ROOT / "data" / "artifacts" / "demo")
    profile = os.environ.get("LITBIORAG_MODEL_PROFILE") or ""
    client_mode = os.environ.get("AEOS_CLIENT_MODE", "0") == "1"
    if profile == "offline-demo":
        _offline_demo(bundle, profile)
    if client_mode:
        _client_mode_guard(bundle, profile)
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

        if client_mode:
            print("[entrypoint] client mode: no demo tokens; issue prospect tokens with "
                  "scripts/issue_client_token.py", flush=True)
        else:
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
