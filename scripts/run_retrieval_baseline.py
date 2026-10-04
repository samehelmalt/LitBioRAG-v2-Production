"""Run the retrieval pipeline over a query set against a bundle and write reports.

    python -m scripts.run_retrieval_baseline --bundle data/artifacts/pubmed2015plus_v1 \
        --encoder hf:ncbi/MedCPT-Query-Encoder --reranker hf:ncbi/MedCPT-Cross-Encoder \
        --queries thesis24 --modes bm25 dense hybrid hybrid+rerank \
        --out evaluation/benchmarks/runs/thesis24_medcpt_v1

--queries: ``thesis24`` (the thesis evaluation set) or a split name from the manifest
(``tuning``, ``validation``, ``calibration``, ``test``). The encoder must match the bundle's.
One sub-directory per mode: rows.jsonl, summary.json, report.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.reranking.cross_encoder import make_reranker  # noqa: E402
from app.retrieval.bm25 import BM25Index  # noqa: E402
from app.retrieval.dense import DenseIndex  # noqa: E402
from app.retrieval.encoders import make_encoder  # noqa: E402
from app.retrieval.pipeline import MODES, HybridRetriever, RetrievalConfig  # noqa: E402
from app.retrieval.store import MetadataStore  # noqa: E402
from evaluation.datasets.loaders import load_all, load_thesis_queries  # noqa: E402
from evaluation.runners.retrieval import run_retrieval, write_run  # noqa: E402
from evaluation.splits import MANIFEST_DIR, SplitManifest  # noqa: E402


def load_records(spec: str, manifest_path: Path):
    if spec == "thesis24":
        return load_thesis_queries()
    manifest = SplitManifest.load(manifest_path)
    ids = set(manifest.ids(spec))
    if not ids:
        raise SystemExit(f"split {spec!r} is empty or unknown in {manifest_path}")
    return [r for r in load_all() if r.id in ids]


def corpus_doc_ids(store: MetadataStore) -> set[str]:
    rows = store.conn.execute("SELECT doc_id FROM documents")
    return {r[0] for r in rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--encoder", required=True, help="must match the bundle (hash:<dim> | hf:<id>)")
    ap.add_argument("--reranker", default="none", help="none | lexical | hf:<model_id>")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "retrieval.yaml")
    ap.add_argument("--queries", default="thesis24", help="thesis24 or a manifest split name")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "manifest_v1.json")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N queries (smoke tests)")
    a = ap.parse_args(argv)

    cfg = RetrievalConfig.from_yaml(a.config)
    store = MetadataStore(a.bundle / "metadata.sqlite", read_only=True)
    bm25 = BM25Index(a.bundle / "bm25")
    dense = DenseIndex(a.bundle / "dense", make_encoder(a.encoder))
    reranker = make_reranker(a.reranker)
    retriever = HybridRetriever(store, bm25, dense, reranker, cfg)
    records = load_records(a.queries, a.manifest)
    if a.limit:
        records = records[: a.limit]
    corpus = corpus_doc_ids(store)
    manifest = json.loads((a.bundle / "manifest.json").read_text())
    meta_base = {
        "bundle": manifest.get("corpus_id"),
        "bundle_counts": manifest.get("counts"),
        "encoder": a.encoder,
        "reranker": a.reranker,
        "queries": a.queries,
        "n_records": len(records),
        "config": cfg.model_dump(),
    }
    for mode in a.modes:
        if mode == "hybrid+rerank" and reranker is None:
            print(f"skip {mode}: no reranker given", file=sys.stderr)
            continue
        rows = run_retrieval(retriever, records, mode=mode, corpus_doc_ids=corpus)
        summary = write_run(rows, a.out / mode.replace("+", "_"), meta={**meta_base, "mode": mode})
        o = summary["overall"]
        print(
            f"{mode:14s} n={o['n_queries']:4d} R@5={_f(o.get('recall@5'))} "
            f"R@10={_f(o.get('recall@10'))} MRR={_f(o['mrr'])} failures={o['failures']}"
        )
    return 0


def _f(v):
    return "—" if v is None else f"{v:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
