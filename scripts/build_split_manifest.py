"""Build evaluation/datasets/splits/manifest_v1.json.

Pins: the 24 thesis evaluation queries -> test; the 30 thesis calibration queries -> calibration.
Everything else from BioASQ + PubMedQA is stratified by dataset x question type with whole
gold-PMID groups kept inside one split. Deterministic for a given seed.

Run from the repository root:  python scripts/build_split_manifest.py
"""

from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.datasets.loaders import (  # noqa: E402
    QueryRecord,
    load_all,
    load_thesis_queries,
    normalize_pmid,
    normalize_question,
)
from evaluation.splits import MANIFEST_DIR, build_manifest  # noqa: E402

CALIB_RUN = ROOT / "evaluation" / "benchmarks" / "thesis_baseline" / "calib" / "calib_1st_run"


def load_calibration_queries() -> list[QueryRecord]:
    """The 30 held-out queries, read from the S7 rows of the first calibration run."""
    seen: dict[str, QueryRecord] = {}
    for path in sorted(glob.glob(str(CALIB_RUN / "thesis_ladder_*.csv"))):
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                q = row.get("Query_Full") or row.get("Query") or ""
                key = normalize_question(q)
                if not key or key in seen:
                    continue
                raw_pmids = row["GT_PMIDs"].split(";")
                pmids = tuple(p for p in (normalize_pmid(x) for x in raw_pmids) if p)
                seen[key] = QueryRecord(
                    id=f"calib30-{len(seen):02d}",
                    question=q,
                    gold_pmids=pmids,
                    dataset=row.get("Dataset", ""),
                    question_type=(row.get("Question_Type") or "unknown").lower(),
                )
    return list(seen.values())


def main() -> int:
    records = load_all()
    by_norm = {normalize_question(r.question): r for r in records}

    pinned: dict[str, str] = {}
    missing: list[str] = []
    for r in load_thesis_queries():
        hit = by_norm.get(normalize_question(r.question))
        if hit is None:
            missing.append(r.question)
        else:
            pinned[hit.id] = "test"
    for r in load_calibration_queries():
        hit = by_norm.get(normalize_question(r.question))
        if hit is None:
            missing.append(r.question)
        else:
            pinned[hit.id] = "calibration"
    if missing:
        print("WARNING: pinned queries not found in BioASQ/PubMedQA pool:")
        for q in missing:
            print("  -", q[:100])

    manifest = build_manifest(
        records,
        pinned=pinned,
        version="v1",
        notes=(
            "BioASQ + PubMedQA (thesis extracted sets, de-duplicated by question text). "
            "Thesis 24 evaluation queries pinned to test; thesis 30 calibration queries pinned to "
            "calibration. Whole gold-PMID groups stay in one split. adversarial is filled by "
            "evaluation/adversarial, not here."
        ),
    )
    out = MANIFEST_DIR / "manifest_v1.json"
    manifest.save(out)
    print(f"wrote {out.relative_to(ROOT)}")
    print("records:", len(records), "pinned:", len(pinned))
    print("counts:", manifest.counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
