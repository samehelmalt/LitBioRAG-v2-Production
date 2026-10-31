"""Run retrieval → evidence → generation → verification → gate end to end and print the result.

    python -m scripts.demo_pipeline --bundle data/artifacts/<id> --profile test \
        --question "What does MAP3K3 do in NF-kappaB signalling?"

Models come from configs/models.yaml (profile) or LITBIORAG_MODEL__<ROLE> overrides; nothing
here names a model. With the `test` profile the whole chain runs without weights. The decision
comes from configs/gate.yaml; until the calibration job has run it is marked UNCALIBRATED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.service.pipeline import AnswerPipeline  # noqa: E402


def run(bundle: Path, question: str, profile: str | None, *, max_units: int = 8) -> dict:
    """Same code path as the API (app/service/pipeline.py); prints the full trust surface."""
    pl = AnswerPipeline(bundle, profile=profile, max_units=max_units)
    out = pl.run(question)
    return {
        "question": question,
        "models": pl.registry.describe(),
        "versions": pl.versions.as_dict(),
        "preflight": pl.preflight,
        "decision": out.decision.model_dump(),
        "gate_calibrated_on": pl.versions.gate_calibrated_on or "UNCALIBRATED (heuristic defaults)",
        "evidence": [
            {"citation": u.citation_id, "rank": u.doc_rank, "peer_reviewed": u.is_peer_reviewed,
             "title": u.title}  # fmt: skip
            for u in out.evidence
        ],
        "answer": out.generated.model_dump(exclude={"raw"}),
        "verification": out.report.model_dump(),
        "timings_ms": out.timings_ms,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--profile", default=None, help="models.yaml profile (default: file default)")
    ap.add_argument("--max-units", type=int, default=8)
    a = ap.parse_args(argv)
    out = run(a.bundle, a.question, a.profile, max_units=a.max_units)
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
