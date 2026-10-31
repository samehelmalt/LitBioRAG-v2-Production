"""Run retrieval → evidence → generation → verification end to end and print the report.

    python -m scripts.demo_pipeline --bundle data/artifacts/<id> --profile test \
        --question "What does MAP3K3 do in NF-kappaB signalling?"

Models come from configs/models.yaml (profile) or LITBIORAG_MODEL__<ROLE> overrides; nothing
here names a model. With the `test` profile the whole chain runs without weights. The gate
(Step 8) is not applied yet: the output is the verification report the gate will consume.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evidence.selector import select_evidence  # noqa: E402
from app.generation.service import GenerationConfig, answer  # noqa: E402
from app.retrieval.bm25 import BM25Index  # noqa: E402
from app.retrieval.dense import DenseIndex  # noqa: E402
from app.retrieval.pipeline import HybridRetriever, RetrievalConfig  # noqa: E402
from app.retrieval.store import MetadataStore  # noqa: E402
from app.verification.pipeline import VerificationConfig, verify  # noqa: E402
from models.registry import ModelRegistry  # noqa: E402


def run(bundle: Path, question: str, profile: str | None, *, max_units: int = 8) -> dict:
    reg = ModelRegistry.load(profile=profile)
    t0 = time.perf_counter()
    store = MetadataStore(bundle / "metadata.sqlite", read_only=True)
    retriever = HybridRetriever(
        store,
        BM25Index(bundle / "bm25"),
        DenseIndex(bundle / "dense", reg.build("dense_encoder")),
        reg.build("reranker"),
        RetrievalConfig.from_yaml(ROOT / "configs" / "retrieval.yaml"),
    )
    res = retriever.retrieve(question, mode="hybrid+rerank")
    units = select_evidence(res, store, max_units=max_units)
    t1 = time.perf_counter()
    gen_cfg = GenerationConfig.from_yaml(ROOT / "configs" / "generation.yaml")
    gen = answer(question, units, reg.build("generator"), config=gen_cfg)
    t2 = time.perf_counter()
    report = verify(
        gen, units, reg.build("verifier"), reg.build("entity_extractor"),
        config=VerificationConfig.from_yaml(ROOT / "configs" / "verification.yaml"),
        thresholds=reg.thresholds("verifier"),
    )  # fmt: skip
    t3 = time.perf_counter()
    return {
        "question": question,
        "models": reg.describe(),
        "evidence": [
            {"citation": u.citation_id, "rank": u.doc_rank, "peer_reviewed": u.is_peer_reviewed,
             "title": u.title}  # fmt: skip
            for u in units
        ],
        "answer": gen.model_dump(exclude={"raw"}),
        "verification": report.model_dump(),
        "timings_ms": {
            "retrieval": round((t1 - t0) * 1000, 1),
            "generation": round((t2 - t1) * 1000, 1),
            "verification": round((t3 - t2) * 1000, 1),
        },
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
