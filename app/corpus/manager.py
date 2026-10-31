"""Corpus index management for the served bundles (PubMed, bioRxiv/medRxiv, decks).

Bundles live under ``artifacts_dir/<corpus_id>/`` with a manifest. The manager lists them,
activates one (the pipeline swaps its retriever in place), runs maintenance jobs in background
threads with a log each, and holds the "indexed corpus enabled" switch (off → every query is
served from uploaded knowledge bases only).

Jobs: ``fetch`` a bundle from Drive/URL/path (artifacts.fetch), ``verify`` checksums, ``update``
(append JSONL documents: scripts/update_bundle), ``fetch_preprints`` (bioRxiv/medRxiv window →
JSONL, then update), ``rebuild`` (full build: refused on the serving host unless
``allow_build_on_serve`` — the UI shows the build-host command instead).
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parents[2]
JOB_TYPES = ("fetch", "verify", "update", "fetch_preprints", "rebuild")


class CorpusConfig(BaseModel):
    artifacts_dir: str = "data/artifacts"
    jobs_dir: str = "data/jobs"
    staging_dir: str = "data/staging"
    allow_build_on_serve: bool = False
    indexed_enabled: bool = True
    encoder_spec: str = ""  # e.g. hf:ncbi/MedCPT-Article-Encoder; "" = profile encoder name

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> CorpusConfig:
        p = Path(path or os.environ.get("LITBIORAG_CORPUS_CONFIG")
                 or _ROOT / "configs" / "corpus.yaml")  # fmt: skip
        return cls.model_validate(yaml.safe_load(p.read_text()) or {}) if p.exists() else cls()


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else _ROOT / q


class CorpusManager:
    def __init__(self, cfg: CorpusConfig | None = None, *,
                 activate: Callable[[Path], dict] | None = None,
                 active_bundle: Path | None = None, encoder_name: str = ""):  # fmt: skip
        self.cfg = cfg or CorpusConfig.from_yaml()
        self.artifacts_dir = _abs(self.cfg.artifacts_dir)
        self.jobs_dir = _abs(self.cfg.jobs_dir)
        self.staging_dir = _abs(self.cfg.staging_dir)
        for d in (self.artifacts_dir, self.jobs_dir, self.staging_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._activate = activate
        self.active_bundle = active_bundle
        self.encoder_name = encoder_name
        self.indexed_enabled = self.cfg.indexed_enabled
        self.jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ---- bundles ---------------------------------------------------------------------------
    def list_bundles(self) -> list[dict[str, Any]]:
        out = []
        candidates = sorted(p for p in self.artifacts_dir.iterdir() if p.is_dir())
        if self.active_bundle and self.active_bundle not in candidates:
            candidates.insert(0, self.active_bundle)
        for p in candidates:
            m = p / "manifest.json"
            if not m.exists():
                continue
            try:
                man = json.loads(m.read_text())
            except ValueError:
                continue
            enc = (man.get("encoder") or {}).get("encoder", "")
            dense = man.get("dense", {})
            active = (self.active_bundle is not None
                      and p.resolve() == self.active_bundle.resolve())  # fmt: skip
            out.append({
                "corpus_id": man.get("corpus_id", p.name), "path": str(p),
                "created_at": man.get("created_at", ""), "counts": man.get("counts", {}),
                "encoder": enc,
                "dense": {k: dense.get(k) for k in ("type", "n", "stale_vectors")},
                "updates": man.get("updates", []),
                "total_mb": round(man.get("total_bytes", 0) / 1e6, 1),
                "active": active,
                "encoder_matches_profile": not self.encoder_name or enc == self.encoder_name,
            })  # fmt: skip
        return out

    def bundle_path(self, corpus_id: str) -> Path:
        for b in self.list_bundles():
            if b["corpus_id"] == corpus_id:
                return Path(b["path"])
        raise LookupError(f"unknown bundle {corpus_id!r}")

    def activate(self, corpus_id: str) -> dict[str, Any]:
        path = self.bundle_path(corpus_id)
        from scripts.artifacts import verify

        problems = verify(path)
        if problems:
            raise ValueError("bundle fails verification: " + "; ".join(problems[:5]))
        info = self._activate(path) if self._activate else {}
        self.active_bundle = path
        return {"active": corpus_id, **info}

    # ---- jobs ------------------------------------------------------------------------------
    def start_job(self, job_type: str, params: dict[str, Any], *, by: str = "") -> dict[str, Any]:
        if job_type not in JOB_TYPES:
            raise ValueError(f"job type must be one of {JOB_TYPES}")
        if job_type == "rebuild" and not self.cfg.allow_build_on_serve:
            raise PermissionError(
                "full rebuilds run on the build host, not the serving host: "
                + self.rebuild_command(params))
        job_id = f"{job_type}-{uuid.uuid4().hex[:8]}"
        log = self.jobs_dir / f"{job_id}.log"
        job = {"job_id": job_id, "type": job_type, "params": params, "by": by,
               "status": "running", "started_at": time.time(), "log": str(log),
               "result": None, "error": None}  # fmt: skip
        self.jobs[job_id] = job
        threading.Thread(target=self._run, args=(job, log), daemon=True).start()
        return job

    def _run(self, job: dict[str, Any], log: Path) -> None:
        def say(msg: str) -> None:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(time.strftime("%H:%M:%S ") + msg + "\n")

        try:
            say(f"start {job['type']} {json.dumps(job['params'])}")
            job["result"] = self._dispatch(job["type"], job["params"], say)
            job["status"] = "done"
            say("done " + json.dumps(job["result"], default=str))
        except Exception as e:  # noqa: BLE001
            job["status"], job["error"] = "failed", f"{type(e).__name__}: {e}"
            say("failed " + job["error"] + "\n" + traceback.format_exc())
        job["finished_at"] = time.time()

    def _dispatch(self, kind: str, p: dict[str, Any], say) -> dict[str, Any]:
        from scripts import artifacts

        if kind == "fetch":
            dest = self.artifacts_dir / _safe(p["corpus_id"])
            problems = artifacts.fetch(p["source"], dest)
            if problems:
                raise RuntimeError("fetched bundle fails verification: " + "; ".join(problems[:5]))
            return {"bundle": str(dest)}
        if kind == "verify":
            problems = artifacts.verify(self.bundle_path(p["corpus_id"]))
            return {"ok": not problems, "problems": problems}
        if kind == "update":
            from scripts.update_bundle import update_bundle

            bundle = self.bundle_path(p["corpus_id"])
            enc = p.get("encoder") or self.cfg.encoder_spec or "hash:128"
            res = update_bundle(bundle, list(p["docs"]), enc)
            is_active = (self.active_bundle is not None
                         and bundle.resolve() == self.active_bundle.resolve())  # fmt: skip
            if is_active and self._activate:
                say("reloading active bundle")
                self._activate(bundle)
            return res
        if kind == "fetch_preprints":
            from scripts.corpus.preprints import fetch_all, write_documents

            dest = self.staging_dir / f"{p['server']}-{p['from']}-{p['to']}"
            dest.mkdir(parents=True, exist_ok=True)
            n = fetch_all(p["server"], p["from"], p["to"], dest, max_pages=p.get("max_pages"))
            info = write_documents(p["server"], dest)
            say(f"fetched {n} records → {info}")
            out = {"staging": str(dest), **(info if isinstance(info, dict) else {})}
            if p.get("corpus_id"):
                docs = [str(dest / f"{p['server']}.jsonl")]
                out["update"] = self._dispatch(
                    "update", {"corpus_id": p["corpus_id"], "docs": docs,
                               "encoder": p.get("encoder")}, say)  # fmt: skip
            return out
        if kind == "rebuild":
            from scripts.build_indexes import main as build

            out = self.artifacts_dir / _safe(p["corpus_id"])
            if out.exists():
                shutil.rmtree(out)
            rc = build(["--docs", *p["docs"], "--out", str(out), "--corpus-id", p["corpus_id"],
                        "--encoder", p.get("encoder") or self.cfg.encoder_spec or "hash:128"])
            if rc != 0:
                raise RuntimeError(f"build_indexes exited {rc}")
            return {"bundle": str(out)}
        raise ValueError(kind)

    def rebuild_command(self, p: dict[str, Any]) -> str:
        docs = " ".join(p.get("docs", ["<docs.jsonl>"]))
        cid = p.get("corpus_id", "<corpus_id>")
        enc = p.get("encoder") or self.cfg.encoder_spec or "<encoder>"
        return (f"python -m scripts.build_indexes --docs {docs} --out artifacts/{cid} "
                f"--corpus-id {cid} --encoder {enc}  # run on the build host")

    def job(self, job_id: str, *, tail: int = 50) -> dict[str, Any] | None:
        j = self.jobs.get(job_id)
        if j is None:
            return None
        lines: list[str] = []
        if Path(j["log"]).exists():
            lines = Path(j["log"]).read_text(encoding="utf-8").splitlines()[-tail:]
        return {**j, "log_tail": lines}

    def status(self) -> dict[str, Any]:
        return {"indexed_enabled": self.indexed_enabled,
                "active": str(self.active_bundle) if self.active_bundle else None,
                "bundles": self.list_bundles(),
                "jobs": [{k: v for k, v in j.items() if k != "log"} for j in self.jobs.values()],
                "rebuild_hint": self.rebuild_command({})}  # fmt: skip


def _safe(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum() or ch in "._-") or "bundle"
