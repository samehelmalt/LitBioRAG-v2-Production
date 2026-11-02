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
| entity_extractor | `regex:biomed-v2` (default), scispaCy `en_core_sci_md` + UMLS linker | |
| slide images | Tesseract OCR; hosted vision model (`configs/kb.yaml`) | image-only slides are flagged until one is enabled |

**MedCPT is a paired encoder**: `ncbi/MedCPT-Article-Encoder` embeds passages and
`ncbi/MedCPT-Query-Encoder` embeds questions. The registry refuses the article tower without its
query tower; every profile names the pair (`params.query`). The thesis used the article tower for
both, which the production baseline will measure as a separate row.

## Profiles for the education prototype
| Profile | Generator | Retrieval / verifier | Fits 12 GB? | Use |
|---|---|---|---|---|
| `local-8b` (A) | Qwen3-8B 4-bit, thinking off | MedCPT pair + bge-reranker-v2-m3 + DeBERTa-v3-large NLI | yes, all resident (~8.7 GB) | first prototype run; private data |
| `local-14b` (B) | Qwen3-14B 4-bit | same | generator on_demand or second GPU | quality comparison on identical inputs |
| `hosted` (C) | any OpenAI-compatible / Anthropic model | same, local | generator uses no VRAM | when confidentiality permits; spend caps apply |

## Prototype benchmark protocol (30 slides, before scaling to 5,000)
1. Build the deck + question set (`scripts/make_prototype_deck.py`) or use the client's reviewed
   set (same schema, `evaluation/datasets/lecture_qa.py`).
2. For each profile, run `scripts/run_lecture_qa.py` with the **same** deck, evidence windows and
   questions; the run directory records versions, question-set summary and the attempt ledger.
3. Compare: slide Recall@5/10 and MRR; citation-resolution validity (must be 100%); delivered
   precision and false PASS (proxy, then human labels in the `human_label` column); unanswerable
   abstention; repetition / format defect rate; attempts per answer; latency; cost per 1,000
   answers. Report candidate / verified / released metrics separately.
4. Choose one winner per role; calibrate the gate on the calibration split for that verifier;
   record the row below with the run id. No row, no claim.

## Results

None yet. Each benchmark run appends a dated row here with the run directory under `evaluation/benchmarks/runs/`.


## Hosted generator profile
`profiles.hosted` in `configs/models.yaml` routes the generator to any OpenAI-compatible endpoint
(or Anthropic) using an API key from the environment variable named in `api_key_env`. Nothing else
changes: the same contract, verifier, gate and ledger apply. The hosted candidates enter the Step 6
benchmark alongside the local 4-bit candidates with cost per 1,000 answers as an extra column
(see COST_AND_RESOURCES.md). The verifier stays a local, different-family model so the
independence rule holds whichever generator is chosen.
