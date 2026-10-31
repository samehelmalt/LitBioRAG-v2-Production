"""Run the full answer pipeline over a query set and write trust metrics (proxy labels).

    python -m scripts.run_answer_baseline --bundle data/artifacts/<id> --profile default \
        --queries calibration --out evaluation/benchmarks/runs/<name>

Models come from the registry profile; the gate config from configs/gate.yaml. Use the
`calibration` split to produce rows for scripts/calibrate_gate.py; report on `validation`; touch
`test` only for the final comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.generation.service import GenerationConfig  # noqa: E402
from app.retrieval.bm25 import BM25Index  # noqa: E402
from app.retrieval.dense import DenseIndex  # noqa: E402
from app.retrieval.pipeline import HybridRetriever, RetrievalConfig  # noqa: E402
from app.retrieval.store import MetadataStore  # noqa: E402
from app.safety.gate import GateConfig  # noqa: E402
from app.verification.pipeline import VerificationConfig  # noqa: E402
from evaluation.runners.answer import run_answers, write_run  # noqa: E402
from evaluation.splits import MANIFEST_DIR  # noqa: E402
from models.registry import ModelRegistry  # noqa: E402
from scripts.run_retrieval_baseline import load_records  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--queries", default="calibration")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "manifest_v1.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    reg = ModelRegistry.load(profile=a.profile)
    store = MetadataStore(a.bundle / "metadata.sqlite", read_only=True)
    retriever = HybridRetriever(
        store, BM25Index(a.bundle / "bm25"),
        DenseIndex(a.bundle / "dense", reg.build("dense_encoder")), reg.build("reranker"),
        RetrievalConfig.from_yaml(ROOT / "configs" / "retrieval.yaml"),
    )  # fmt: skip
    records = load_records(a.queries, a.manifest)
    if a.limit:
        records = records[: a.limit]
    gate_cfg = GateConfig.from_yaml(ROOT / "configs" / "gate.yaml")
    rows = run_answers(
        retriever, records,
        generator=reg.build("generator"), verifier=reg.build("verifier"),
        extractor=reg.build("entity_extractor"),
        gen_cfg=GenerationConfig.from_yaml(ROOT / "configs" / "generation.yaml"),
        ver_cfg=VerificationConfig.from_yaml(ROOT / "configs" / "verification.yaml"),
        gate_cfg=gate_cfg, thresholds=reg.thresholds("verifier"),
    )  # fmt: skip
    meta = {"models": reg.describe(), "queries": a.queries, "n": len(records)}
    s = write_run(rows, a.out, meta=meta, calibrated_on=gate_cfg.calibrated_on)
    g = s.get("gate", {})
    print(f"n={s['n']} actions={s['actions']} false_pass={g.get('false_pass_rate')} "
          f"coverage={g.get('coverage')} ({s['calibrated_on']})")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
