# Model Selection

**Status: no model has been selected yet.** Every entry in `configs/models.yaml` profile `default` is a *candidate* sized for the stated hardware (12 GB VRAM, 16 GB RAM, quantized). A candidate becomes the selected model only when a benchmark run on the harness is recorded here with its numbers, and no earlier than Steps 5–7.

## How a model is chosen

For every role, one benchmark per role, one variable at a time, on the `validation` split (never `test`):

| Role | Benchmark | Decides on |
|---|---|---|
| dense_encoder | `scripts/run_retrieval_baseline.py --modes dense hybrid` with each candidate bundle | recall@100 of the candidate pool, recall@10 after reranking, encode throughput, index size |
| reranker | same, `--modes hybrid+rerank`, one reranker per run | recall@5/10, MRR, nDCG@10, ms per 100 pairs, VRAM |
| generator | Step 6 runner | claim entailment rate, unsupported rate, abstention compliance, citation validity, yes/no accuracy, tokens/s, VRAM, bf16 → 4-bit delta |
| verifier | Step 7 runner | precision/recall on proxy-labelled claims, adversarial catch rate (swapped entity, altered number, negation flip), ms per pair; **must be a different family from the generator** |
| entity_extractor | Step 7 runner | entity F1 on a labelled sample, retrieval effect of entity-aware expansion |

A candidate is **rejected** if it raises false PASS (or the proxy for it) regardless of gains elsewhere (`docs/PRODUCTION_AUDIT.md` §35 regression policy).

## How to swap a model

Models are configuration, not code: edit `configs/models.yaml` or set `LITBIORAG_MODEL__<ROLE>`. See `models/registry/README.md`. Two consequences the registry enforces:

- a swapped **verifier, reranker or generator** loses its calibrated thresholds until the calibration job records new ones for the new model id;
- a swapped **dense encoder** requires re-embedding the bundle (`scripts/build_indexes.py` resumes from the embed stage after the old shards are removed).

## Candidates under consideration (not decisions)

| Role | Candidates | Notes |
|---|---|---|
| dense_encoder | `ncbi/MedCPT-Article-Encoder` (thesis-compatible baseline), `Qwen/Qwen3-Embedding-0.6B`, `BAAI/bge-m3`, `BMRetriever-410M` | thesis showed the choice matters little after a strong reranker; pick for pool recall and speed |
| reranker | `ncbi/MedCPT-Cross-Encoder` (baseline), `BAAI/bge-reranker-v2-m3`, `Qwen/Qwen3-Reranker-0.6B`, 4-bit `Qwen/Qwen3-Reranker-4B` | single stage over the full pool, full token length |
| generator (local, ≤ 12 GB at 4-bit) | `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `google/gemma-3-12b-it`, `google/medgemma-4b-it`, `meta-llama/Llama-3.1-8B-Instruct` | structured-output compliance weighs more than chain-of-thought (thesis E1) |
| generator (hosted, optional) | any 27B-class or API model behind `openai_compatible` / `anthropic` | cost and latency columns added |
| verifier | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, Bespoke-MiniCheck-7B (4-bit), `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (thesis baseline) | different family from the generator |
| entity_extractor | `regex:biomed-v1` (default), scispaCy `en_core_sci_md` + UMLS linker | |

## Results

None yet. Each benchmark run appends a dated row here with the run directory under `evaluation/benchmarks/runs/`.
