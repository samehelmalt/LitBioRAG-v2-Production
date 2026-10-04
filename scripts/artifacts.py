"""Artifact bundles: pack on the build host, verify and fetch on the serving host.

    python -m scripts.artifacts pack   artifacts/pubmed2015plus_v1 --corpus-id pubmed2015plus_v1
    python -m scripts.artifacts verify artifacts/pubmed2015plus_v1
    python -m scripts.artifacts fetch  <source> artifacts/pubmed2015plus_v1

``pack`` writes ``manifest.json`` with a sha256 and byte size for every file in the directory.
``verify`` recomputes them and exits non-zero on any mismatch or missing file.
``fetch`` copies a bundle from a local path (for example a mounted or rclone-synced Google Drive
folder) or from an https URL that serves the bundle directory, then verifies it. Google Drive
itself is reached with rclone or the Drive desktop client; this script never talks to the Drive
API. Partial transfers are safe: files whose hash already matches are not copied again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

MANIFEST = "manifest.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(bundle: Path, *, corpus_id: str, extra: dict | None = None) -> dict:
    files = []
    for p in sorted(bundle.rglob("*")):
        if p.is_file() and p.name != MANIFEST and not p.name.endswith(".part"):
            rel = p.relative_to(bundle).as_posix()
            files.append({"path": rel, "sha256": sha256_of(p), "bytes": p.stat().st_size})
    manifest = {
        "corpus_id": corpus_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
        **(extra or {}),
    }
    (bundle / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def verify(bundle: Path) -> list[str]:
    """Return a list of problems (empty when the bundle is intact)."""
    mpath = bundle / MANIFEST
    if not mpath.exists():
        return ["manifest.json missing"]
    manifest = json.loads(mpath.read_text())
    problems = []
    for f in manifest["files"]:
        p = bundle / f["path"]
        if not p.exists():
            problems.append(f"missing: {f['path']}")
        elif p.stat().st_size != f["bytes"]:
            problems.append(f"size mismatch: {f['path']}")
        elif sha256_of(p) != f["sha256"]:
            problems.append(f"sha256 mismatch: {f['path']}")
    return problems


def _read_remote(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=600) as r:
        return r.read()


def fetch(source: str, dest: Path) -> list[str]:
    """Copy a bundle from a local directory or an https base URL into ``dest``; then verify."""
    dest.mkdir(parents=True, exist_ok=True)
    is_url = source.startswith(("http://", "https://"))
    base = source.rstrip("/")
    if is_url:
        manifest = json.loads(_read_remote(f"{base}/{MANIFEST}").decode())
    else:
        manifest = json.loads((Path(source) / MANIFEST).read_text())
    for f in manifest["files"]:
        target = dest / f["path"]
        if target.exists() and target.stat().st_size == f["bytes"]:
            if sha256_of(target) == f["sha256"]:
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        if is_url:
            tmp.write_bytes(_read_remote(f"{base}/{f['path']}"))
        else:
            shutil.copyfile(Path(source) / f["path"], tmp)
        tmp.replace(target)
    (dest / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    return verify(dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("bundle", type=Path)
    p.add_argument("--corpus-id", required=True)
    p = sub.add_parser("verify")
    p.add_argument("bundle", type=Path)
    p = sub.add_parser("fetch")
    p.add_argument("source")
    p.add_argument("dest", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "pack":
        m = pack(a.bundle, corpus_id=a.corpus_id)
        print(f"packed {len(m['files'])} files, {m['total_bytes']/1e6:.1f} MB")
        return 0
    problems = verify(a.bundle) if a.cmd == "verify" else fetch(a.source, a.dest)
    for pr in problems:
        print(pr, file=sys.stderr)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
