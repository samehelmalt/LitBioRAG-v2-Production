# Cost and resources: tokens, API spend, GPUs

Owner decisions (2026-10-31): the generator may run behind an API key; everything must be
optimized for tokens, spend and the 12 GB-VRAM / 16 GB-RAM serving hosts. This document describes
the controls that exist in code and what is still to be measured. No number below is a measurement
until BASELINE_RESULTS.md says so.

## Where the knobs are
| Concern | File | What it controls |
|---|---|---|
| Model choice, API endpoint and key variable | `configs/models.yaml` (profile `hosted`) or `LITBIORAG_MODEL__GENERATOR=openai_compatible:<model>?endpoint=<url>&api_key_env=<VAR>` | which model answers; keys are read from the environment only, never from config or UI |
| Token budget for the prompt | `configs/resources.yaml: budgets.generator` | context window, output reserve, evidence token cap, per-passage cap |
| Spend caps and retries | `configs/resources.yaml: api` | per-request and per-day USD caps, timeout, retry count and backoff |
| Prices | `configs/pricing.yaml` | USD per 1M tokens per model id; `as_of` date must be filled before a budget is trusted |
| GPU residency | `configs/resources.yaml: gpu` | which roles stay loaded, which are loaded per request, VRAM estimates for the pre-flight check |
| Caching | `configs/resources.yaml: cache` | on-disk cache for verifier pair scores and query embeddings |
| Admission control | `configs/resources.yaml: concurrency` (`AEOS_CONCURRENCY_<ROLE>` override) | how many requests may hold the generator / verifier at once, queue bound and timeout; overflow → 503 + Retry-After |
| Micro-batching | `configs/resources.yaml: batching` | concurrent verifier / reranker calls share one forward pass (collection window, batch size) |
| Shared state | `configs/api.yaml: state_backend`, `REDIS_URL` | Redis-backed rate limiter, trial / usage ledgers and KB jobs for several API replicas |

## Token optimization (`app/core/budget.py`)
1. **Evidence is trimmed to the budget in rank order.** `fit_evidence` keeps the best-ranked units
   while they fit `evidence_tokens = min(max_evidence_tokens, context_window − reserve_output −
   prompt_overhead)`. A passage longer than `max_tokens_per_unit` is cut at a sentence boundary
   and its `char_end` is adjusted, so citations still point into the real source text.
2. **Counting is exact when possible.** `HFTokenCounter` uses the generator's tokenizer; otherwise
   `HeuristicCounter` over-estimates slightly on purpose (overflow is worse than one dropped
   passage). Every answer carries `evidence_fit` (kept / truncated / dropped passage ids) and
   `usage` (prompt and completion tokens, exact or estimated, latency, cost).
3. **Verifier pairs are bounded** (`budgets.verifier.max_pairs`) and premises chunked at the
   model's token limit (`HFNLIVerifier`), with max-entailment across chunks.
4. **The prompt is versioned** (`answer-json-v1`); `prompt_overhead` is re-measured when it changes.

5. **Evidence compression** (`app/evidence/compress.py`, `generation.yaml: compress_evidence`):
   the generator sees the question-relevant window of each passage (≤ `max_sentences_per_unit`
   sentences, offsets preserved); the verifier still sees the full passage. `evidence_fit`
   reports `evidence_tokens_full` vs `evidence_tokens_sent` on every answer.
6. **Compact schema** in the prompt (~250 instead of ~420 tokens) and **generation cache**
   keyed by model + prompt id + question + exact evidence spans (`resources.yaml: cache`), so a
   repeated question costs zero generator tokens and a changed bundle never serves a stale answer.
   See QUALITY_PLAYBOOK.md §5 for the full table.

## Spend control (`UsageLedger`)
- Before each hosted call the service estimates the cost (prompt tokens + `max_tokens` output at
  the configured price) and asks the ledger whether it is affordable under the per-request and
  per-day caps. If not, the answer is an **abstention with reason `resource_limit`**: never a
  silent failure, never a truncated answer.
- After each call the real usage (provider-reported when available) is appended to
  `data/usage/ledger.jsonl` with request id, role, model, tokens, cost and the pricing date. The
  ledger replays today's spend on restart.
- Hosted adapters retry on 408/409/425/429/5xx with exponential backoff and give up after
  `max_retries`; the failure then surfaces as an error in the API layer (Step 9), not as an answer.

## GPU plan (`models/registry/pool.py`)
Serving host: 12 GB VRAM per GPU, 16 GB RAM. Rough weights at the configured precision:

| Role | Candidate | Precision | Est. VRAM |
|---|---|---|---|
| dense encoder (query only at serve time) | MedCPT-Query-Encoder (110M) | fp16 | 0.5 GB |
| reranker | bge-reranker-v2-m3 (568M) | fp16 | 1.2 GB |
| verifier | DeBERTa-v3-large NLI (435M) | fp16 | 1.0 GB |
| generator | Qwen3-8B | 4-bit + KV cache | ~6 GB |
| generator | Qwen3-14B | 4-bit + KV cache | ~9.5 GB (does not fit beside the others) |
| generator | 27B class | 4-bit | ~15–16 GB weights alone: hosted/API only |

Policy: encoder, reranker and verifier are `resident`; the local generator is `on_demand` (loaded
for the call, freed afterwards) unless a second GPU is present, in which case it is pinned there
via `device: cuda:1` in `models.yaml`. `ModelPool.preflight()` sums the estimates and refuses a
profile that cannot fit instead of failing on the first request. With the `hosted` profile the
generator uses no VRAM and the whole stack fits with room to spare; this is the recommended
production shape while the local generator benchmark (Step 6) is pending.

## Caching (`app/core/cache.py`)
Verifier (premise, hypothesis) scores and query embeddings are cached on disk keyed by model name,
so a model swap never serves stale scores. Generations are not cached by default: the evidence set
changes with the corpus bundle. Hit rates are logged and will be reported in BASELINE_RESULTS.md.

## Concurrency (`app/core/admission.py`, `app/core/batcher.py`)
An in-process generator serves one sequence at a time, so concurrent requests are admitted per
role: `capacity` holders, a FIFO queue of at most `queue_max` waiters (position shown as the
`queued` progress stage), `queue_timeout_s` before a 503. With vLLM or a hosted generator the
generator capacity is the server's batch size (`AEOS_CONCURRENCY_GENERATOR`). Verifier and
reranker calls from different requests are collected for ≤ `max_wait_ms` and scored in one pass;
a lone caller never waits, and scores are identical to direct calls (tests compare them). Counters
for both are in `/health` (`concurrency`) and in the load-test report. See DEPLOYMENT.md "Scaling".

## What must be measured (E4, blocked on owner resources)
- Tokens per answer (prompt / completion), cost per 1,000 answers for each hosted candidate.
- p50 / p95 latency per stage (retrieval, rerank, generation, verification) on the serving host.
- Cache hit rate on the evaluation splits.
- VRAM peak per profile (`ModelPool.vram_report()`), compared with the pre-flight estimate.
- Throughput and p95 under 10 / 50 / 100 concurrent users (`scripts/load_test.py`) per deployment shape.
