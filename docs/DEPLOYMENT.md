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
| `compose.client-trial.yaml` | client prospect trial: real profile (`PROFILE`), real bundle (`CORPUS_ID`), `configs/api.client-trial.yaml` (authenticated only, hashed questions, education scope, 5 questions per prospect), bound to 127.0.0.1 behind an HTTPS proxy | `AEOS_CLIENT_MODE=1` refuses the `test` profile, the demo bundle and a short/per-run JWT secret; prospect tokens via `scripts/issue_client_token.py` |

OCR in the image: build with `--build-arg WITH_OCR=1` (installs Tesseract and the `ocr` extra);
`configs/kb.yaml: images.ocr: tesseract` enables it. Cloud generator profiles (`hosted`,
`cloud-anthropic`, `cloud-deepseek`) keep the biomedical retrieval and the large local verifier;
selecting a cloud generator never downgrades retrieval or verification. Sending questions and
evidence excerpts to a provider requires the content owner's authorization.
