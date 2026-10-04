# Baseline Results

**Status: NOT YET MEASURED.** No production component has been run. This file defines what the production baseline (Step 4) reports and holds the thesis reference numbers it will be compared against. No improvement over the thesis is claimed.

## Reference: thesis master run (quoted from `docs/THESIS_BASELINE.md` §10)

| | Thesis value | Notes |
|---|---|---|
| R@5 (corpus-normalized), S3a one-stage CE | 0.660 | 24 queries, 2015–2016 corpus |
| R@5, S3 two-stage (path used by the final system) | 0.583 | |
| R@10, one-stage CE | 0.681 | thesis §4.3.4 |
| MRR_norm, S3a | 0.566 | |
| S7 gate split | PASS 14.8% / WARN 34.0% / BLOCK 51.2% | 432 rows |
| S7 median latency | 64.0 s (mean 75.1 s, max 243.8 s) | Colab L4 |
| False-PASS proxies on S7 PASS rows | yes/no wrong 56% (14/25); gold doc not retrieved 34% (22/64) | recomputed for the audit |

## What the production baseline will report

**Retrieval**: Recall@1/5/10/20, MRR, nDCG@10, first relevant rank distribution, per query type and per dataset; candidate-pool recall@100 (what the reranker can possibly recover).

**Generation**: claim entailment rate, unsupported-claim rate, contradiction rate, citation correctness (resolved ∧ supporting), numeric fidelity, entity fidelity, yes/no accuracy where applicable.

**Trust**: false PASS, false WARN, false BLOCK against proxy labels (definitions in `docs/TRUST_AND_SAFETY.md`), abstention precision/recall, ECE, Brier score, coverage–risk curve.

**Performance**: p50/p95/p99 end-to-end and per stage (retrieval, rerank, generation, verification), peak VRAM, peak RAM, queries per minute, on the stated hardware (12 GB VRAM GPUs, 16 GB RAM).

## Protocol

1. Reproduce the thesis ladder numbers from the imported CSVs through the harness (sanity check of the metrics code).
2. Run the clean reimplementation of the thesis algorithms (BM25 + dense + RRF + single-stage CE at full length, thesis generator Qwen2.5-3B as a control) on the thesis 24 queries and the 2015–2016 corpus → "production baseline, thesis-equivalent".
3. Run the same on the v2 validation split and corpus → "production baseline, v2 corpus".
4. Every later change is reported as baseline vs candidate on the same splits; a candidate that raises false PASS is rejected regardless of other gains.
