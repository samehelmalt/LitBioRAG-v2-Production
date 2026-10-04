# Evaluation

The harness in `evaluation/` is the only place where quality is decided. It runs offline, never shares code with the runtime path, and never uses gold labels at inference time. Every production change is reported as **baseline vs candidate** on the same split.

## Data

| Set | Source | Records | Notes |
|---|---|---|---|
| BioASQ (thesis extraction) | `evaluation/datasets/thesis/bioasq_deduped.json` | 1,562 | gold PMIDs; yes/no labels where applicable |
| PubMedQA (thesis extraction) | `evaluation/datasets/thesis/pubmed_deduped.json` | 2,000 | gold PMIDs |
| Thesis 24 | `benchmark_24_queries.json` | 24 | the thesis evaluation set; pinned to `test` |
| Thesis 30 | calibration run 1 CSVs | 30 | the thesis held-out calibration set; pinned to `calibration` |

Loaders: `evaluation/datasets/loaders.py`. Records carry a stable id derived from dataset + normalized question text.

## Splits (`evaluation/datasets/splits/manifest_v1.json`)

Built by `scripts/build_split_manifest.py` (seed 20261004), stratified by dataset × question type.

| Split | Use | n |
|---|---|---|
| tuning | prompt and parameter changes, RRF k, reranker pool sizes | 1,404 |
| validation | model selection between candidates | 704 |
| calibration | gate thresholds / calibrated confidence only | 554 |
| test | reported numbers; touched only for final comparison | 900 |
| adversarial | hand-built in `evaluation/adversarial/` (injection, fabricated PMIDs, swapped entities, altered numbers, conflicting and near-neighbour papers) | — |

Hard rules, enforced by `evaluation/splits.verify`: no normalized question text and no gold PMID appears in two splits. Records sharing a gold PMID are assigned as a group. Do not tune on `test`; do not pick thresholds anywhere but `calibration`.

## Metrics (`evaluation/metrics/`)

**Retrieval** (`retrieval.py`): recall@k, MRR, nDCG@k, recall@1, first relevant rank, and corpus-normalized variants (gold restricted to documents present in the corpus). Same semantics as the thesis, so the imported CSVs are comparable.

**Generation** (`claims.py`): from per-claim verdicts — entailment rate, unsupported rate, contradiction rate, count of contradicted *critical* claims, citation resolution rate (cited ids that exist in the retrieved set), citation support rate (cited ids whose evidence entails the claim), uncited-claim rate, numeric fidelity, entity fidelity.

**Trust** (`trust.py`): false PASS (primary safety metric), false WARN, false BLOCK, unacceptable-released rate, abstention precision/recall, coverage, Brier, ECE (10 equal-width bins), coverage–risk curve and risk at a given coverage.

**Performance** (`performance.py`): p50/p95/p99/mean/max latency; VRAM and RAM are recorded by the runner.

## Labels (`evaluation/labels.py`)

Human faithfulness labels do not exist yet (owner decision 2026-10-04). Until they do, trust metrics use the **proxy label**: an answer is acceptable when (a) yes/no polarity matches gold where applicable, (b) at least one gold document was retrieved, (c) unsupported-claim rate ≤ 0.25, (d) no contradicted claim. These are necessary, not sufficient, conditions; proxy-based false-PASS rates are lower bounds and are always labelled "proxy" in reports. `LabelRecord` / JSONL export and import are in place so that human ratings replace the proxy without code changes elsewhere.

## Thesis baseline import (`evaluation/thesis_import.py`)

`python -m evaluation.thesis_import --report` reproduces the ladder table, the S7 gate distribution and effect, the false-PASS proxies, and re-implements the thesis gate policy from the six logged signals (composite within 1e-3 of the logged value; ≥ 98% decision agreement). `tests/regression/test_thesis_ladder.py` pins these numbers; if the harness ever disagrees with `docs/THESIS_BASELINE.md`, the test fails.

## How to run

```bash
pip install -e ".[dev,eval]"
python scripts/build_split_manifest.py          # regenerates manifest_v1.json deterministically
python -m evaluation.thesis_import --report     # thesis numbers from the imported CSVs
pytest tests/unit tests/regression
```

## What is not yet here

Runners that execute production components against the splits (Step 4), adversarial set construction (Step 7), calibration fitting (Step 8), and resource probes. Each arrives with its own tests and a BASELINE_RESULTS.md update.
