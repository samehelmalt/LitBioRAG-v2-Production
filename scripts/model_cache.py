"""Prefetch the real Hugging Face weights of a profile into a persistent cache, or verify them.

    python -m scripts.model_cache --profile offline-demo            # first run: downloads
    python -m scripts.model_cache --profile offline-demo --offline  # later runs: verify only

Snapshots live under HF_HOME (docker-compose mounts ./data/model-cache). A partial download is
never reported as complete; with --offline a missing snapshot fails clearly instead of reaching
the network. Test doubles and hosted adapters have no weights and are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.registry.loader import ModelRegistry  # noqa: E402

LOCAL_KINDS = ("sentence_transformers", "hf_seq_cls", "hf_causal")


def ensure_models(profile: str, *, offline: bool = False) -> dict:
    from huggingface_hub import snapshot_download  # part of the ml extra

    reg = ModelRegistry.load(profile=profile)
    cache = Path(os.environ.get("HF_HOME", ROOT / "data" / "model-cache")).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for role in reg.roles():
        spec = reg.spec(role)
        if spec.kind not in LOCAL_KINDS:
            results[role] = {"model_id": spec.model_id, "kind": spec.kind, "weights": "none"}
            continue
        ids = [spec.model_id]
        q = spec.params.get("query") or spec.params.get("query_model_id")
        if q:
            ids.append(q)  # paired query tower (MedCPT)
        paths = []
        for mid in ids:
            paths.append(snapshot_download(repo_id=mid, revision=spec.revision,
                                           local_files_only=offline))  # fmt: skip
            print(f"[model-cache] {role}: {mid} -> {paths[-1]}", flush=True)
        results[role] = {"model_id": spec.model_id, "snapshots": paths}
    (cache / f"prefetch-{profile}.json").write_text(json.dumps(results, indent=1))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args(argv)
    ensure_models(a.profile, offline=a.offline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
