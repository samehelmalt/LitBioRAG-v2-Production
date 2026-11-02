# Deployment

Two machine roles. Nothing is built on the serving machine.

| Role | What it needs | What runs there |
|---|---|---|
| **BUILD host** | disk for raw PubMed XML (~40 GB for 2015+) and the bundle (tens of GB), RAM to hold a FAISS training sample, a GPU for embedding | `scripts/corpus/*`, `scripts/build_indexes.py`, `scripts/artifacts.py pack` |
| **SERVE host** | GPUs with 12 GB VRAM, 16 GB system RAM, disk for the bundle | `scripts/artifacts.py fetch/verify`, the API, retrieval, reranking, verification, generation |

## Transfer via Google Drive

The bundle is a plain directory with a `manifest.json` of sha256 hashes, so any file copy works. With rclone:

```bash
# build host
python -m scripts.artifacts pack artifacts/pubmed2015plus_v1 --corpus-id pubmed2015plus_v1
rclone copy artifacts/pubmed2015plus_v1 gdrive:LitBioRAG/artifacts/pubmed2015plus_v1 --progress

# serving host
rclone copy gdrive:LitBioRAG/artifacts/pubmed2015plus_v1 /srv/litbiorag/artifacts/pubmed2015plus_v1 --progress
python -m scripts.artifacts verify /srv/litbiorag/artifacts/pubmed2015plus_v1
```

Or, with the Drive folder mounted locally, `python -m scripts.artifacts fetch /mnt/gdrive/.../pubmed2015plus_v1 /srv/litbiorag/artifacts/pubmed2015plus_v1` copies only files whose hash differs and verifies the result. `fetch` also accepts an https base URL that serves the bundle directory. The scripts never call the Drive API; credentials stay with rclone or the Drive client.

Set `LITBIORAG_DATA_DIR` on the serving host to the directory that contains `artifacts/`; the API loads `metadata.sqlite` read-only, the Tantivy index from disk and the FAISS index memory-mapped.

## Memory budget on the serving host (16 GB RAM, 12 GB VRAM)

- Metadata store and BM25 index: on disk; page cache only.
- FAISS IVF-PQ at m = 64 bytes per vector: ≈ 64 B × passages. 20 M passages ≈ 1.3 GB on disk, memory-mapped. Exact (FlatIP) indexes are used only for small corpora.
- Models (GPU): the generator (≤ 14B at 4-bit, ≤ 9 GB), reranker (0.6B, ≈ 1.5 GB) and verifier cannot all be resident with a KV cache in 12 GB. Step 11 measures sequential load/unload versus pinning the reranker and verifier on a second GPU.

## Status

API, Docker image and compose file are added in Step 9. This document is updated with measured figures (bundle size, load time, RSS, VRAM) after the first real bundle is built.


## Resource controls
Token budgets, API spend caps, GPU residency and caching are configured in `configs/resources.yaml`
and described in [COST_AND_RESOURCES.md](COST_AND_RESOURCES.md). The `hosted` profile in
`configs/models.yaml` runs the generator behind an API key with no generator VRAM; run
`ModelPool.preflight()` (done by the service at start-up in Step 9) before serving a new profile.


## Deployment shapes (compose files)
| File | Purpose | Guard rails |
|---|---|---|
| `docker-compose.yml` | local test run, zero preparation, demo bundle + test doubles | demo tokens printed; never for clients |
| `compose.offline-demo.yaml` | real small CPU models (`offline-demo` profile) on a VPS without a GPU; first start downloads weights into `./data/model-cache` and builds a real-embedding bundle from the synthetic 30-slide deck; later starts work with `AEOS_OFFLINE=1` | profile is NOT medically reliable or calibrated; shows real generation, not accuracy |
| `compose.gpu-vllm.yaml` | one GPU host for many concurrent users: vLLM serves the generator (`vllm-8b` profile, continuous batching), the API keeps retrieval + reranker + verifier on the same GPU, Redis holds the shared state | localhost-bound; `CORPUS_ID` required; vLLM not published |
| `compose.scale.yaml` | several API replicas behind nginx with Redis-shared rate limiter, ledgers and KB jobs; generator = hosted provider or a vLLM host (`VLLM_URL`) | one host, one `./data` volume (SQLite WAL); `PROFILE` and `CORPUS_ID` required |
| `compose.client-trial.yaml` | client prospect trial: real profile (`PROFILE`), real bundle (`CORPUS_ID`), `configs/api.client-trial.yaml` (authenticated only, hashed questions, education scope, 5 questions per prospect), bound to 127.0.0.1 behind an HTTPS proxy | `AEOS_CLIENT_MODE=1` refuses the `test` profile, the demo bundle and a short/per-run JWT secret; prospect tokens via `scripts/issue_client_token.py` |

OCR in the image: build with `--build-arg WITH_OCR=1` (installs Tesseract and the `ocr` extra);
`configs/kb.yaml: images.ocr: tesseract` enables it. Cloud generator profiles (`hosted`,
`cloud-anthropic`, `cloud-deepseek`) keep the biomedical retrieval and the large local verifier;
selecting a cloud generator never downgrades retrieval or verification. Sending questions and
evidence excerpts to a provider requires the content owner's authorization.

## Scaling: many students at the same time

What one API process does under concurrent load, and what the v0.11.0 pieces add:

| Concern | Before | Now |
|---|---|---|
| Local generator (`hf_causal`) | one sequence at a time, no lock: concurrent calls contend or run out of GPU memory | **admission control** (`configs/resources.yaml: concurrency`): per-role capacity, FIFO queue with position on the progress stream (`queued`), `503` + `Retry-After` when the queue is full or the wait exceeds the timeout; counters in `/health.concurrency` |
| Verifier and reranker | one forward pass per request | **cross-request micro-batching** (`batching`): concurrent requests share a pass (≤ 20 ms collection window, identical scores); a lone request takes the direct path |
| Rate limiter, trial ledger, usage ledger, KB jobs | per process | **Redis variants** (`configs/api.yaml: state_backend: redis`, `REDIS_URL`) behind the same interfaces; several replicas enforce one limit and one daily cap |
| Generator throughput | bounded by the single in-process model | **vLLM** as its own service (`vllm-8b` profile, `compose.gpu-vllm.yaml`): continuous batching and paged KV cache; admission capacity set to vLLM's `--max-num-seqs` via `AEOS_CONCURRENCY_GENERATOR` |
| Measurement | none | `scripts/load_test.py` (10/50/100 users, p50/p95/p99, status mix, cache hits, tokens/s) → `evaluation/benchmarks/runs/<name>/load/` |

### Do we need vLLM?
For a handful of concurrent users a local `hf_causal` generator with admission control is enough
(queued answers, no crashes). For hundreds of students asking at the same time, yes: either vLLM
(self-hosted) or a hosted provider, both reached through the `openai_compatible` adapter, so the
switch is a profile change. vLLM does not help the reranker and verifier (micro-batching does) and
does not replace admission control or shared state.

### Capacity estimates (NOT MEASURED; one answer ≈ 3,000 prompt + 500 output tokens)
| Setup | Concurrent generations | Answers / min | Note |
|---|---|---|---|
| `hf_causal` Qwen3-8B 4-bit, 12 GB | 1 | 3–5 | serialized; 15–25 s per answer |
| vLLM Qwen3-8B AWQ, 12 GB shared with reranker + verifier | ~6 | ~20–30 | `VLLM_GPU_FRACTION=0.70` → ~2.9 GB KV ≈ 20k tokens |
| vLLM 8B AWQ, 24 GB (L4 / A10) | ~40–60 | ~120–180 | realistic single-GPU production unit |
| vLLM 8B, 48 GB (L40S / A6000) | ~120–150 | ~350 | one card per few hundred active students |
| hosted provider | provider limit | no GPU | cost per answer; content leaves the host (owner authorization) |

"Thousands of students" normally means 50–300 questions in flight at exam-time peaks, not
thousands at once. The generation cache (same question, same evidence → no generator call) and
short slide evidence cut the load further. Replace this table with the numbers from
`scripts/load_test.py` on the serving host before quoting capacity to a client.

### Run the load test
```bash
TOKEN=$(python -m scripts.issue_client_token --subject loadtest-1 --days 1)
python -m scripts.load_test --base-url http://127.0.0.1:8471 --token "$TOKEN" \
    --users 10 50 100 --requests-per-user 5 --questions data/prototype/questions.jsonl \
    --repeat-ratio 0.5 --out evaluation/benchmarks/runs/$(date +%Y%m%d)-vllm8b/load
```
Read `report.md`: a 503 count is admission control working (clients retry after `Retry-After`);
rising p95 with flat answers/min means the generator is saturated (raise `VLLM_MAX_SEQS` if KV
memory allows, or add a GPU); 429s are the per-principal rate limit, not capacity.

### Shared state and its limits
Redis covers the rate limiter, the trial and usage ledgers and the KB job table. Answer records,
chats, the review queue and the KB files/index stay on one host (`compose.scale.yaml` mounts one
`./data` for all replicas; SQLite in WAL mode handles several writers on one filesystem). Spreading
replicas across hosts needs PostgreSQL for those stores and shared object storage for KB files and
indexes; not built. The admission queue is per replica by design: each replica owns its model slots.
