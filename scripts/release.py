"""Release: run the checks, write dist/aeos-medical-<version>.zip (source only), print a summary.

    python -m scripts.release --version 0.9.0 [--skip-tests]

Steps: ruff → pytest (set -o pipefail semantics: a failing step aborts) → git archive of HEAD
(no data, no secrets, no artifacts) → sha256 of the zip → CHANGELOG check. A GitHub release
is created by the owner from the zip (or `gh release create`), never automatically here.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--version", required=True)
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--screenshots-from", default="",
                    help="base URL of a running demo server to refresh app/web/static/shots")
    a = ap.parse_args(argv)
    if a.version not in (ROOT / "CHANGELOG.md").read_text():
        print(f"CHANGELOG.md has no entry for {a.version}", file=sys.stderr)
        return 2
    if not a.skip_tests:
        run([sys.executable, "-m", "ruff", "check", "."])
        run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
    if a.screenshots_from:
        # regenerate the landing-page screenshots from a running demo server (needs Node Playwright
        # and LITBIORAG_JWT_SECRET); without them the committed PNGs are kept
        subprocess.run([sys.executable, "-m", "scripts.screenshots", "--base-url",
                        a.screenshots_from], cwd=ROOT)  # fmt: skip
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    out = dist / f"aeos-medical-{a.version}.zip"
    run(["git", "archive", "--format=zip", f"--prefix=aeos-medical-{a.version}/", "-o", str(out),
         "HEAD"])  # fmt: skip
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (dist / f"aeos-medical-{a.version}.sha256").write_text(f"{digest}  {out.name}\n")
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB) sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
