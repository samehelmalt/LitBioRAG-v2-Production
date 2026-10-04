# Repository Boundary

```text
SOURCE:
samehelmalt/LitBioRAG
READ ONLY

TARGET:
samehelmalt/LitBioRAG-v2-Production
READ/WRITE
```

## Rules

`samehelmalt/LitBioRAG` is the MSc thesis implementation (engine v5.5.21, master run of 12–13 June 2026). It is a historical reference and is immutable.

Allowed on the source: inspect files, search code, read configuration, read benchmark data and result CSVs, understand algorithms, reuse concepts, copy reviewed source code into this repository, compare thesis and production behaviour.

Not allowed on the source: edit, create, delete or rename files; create branches, commits or pull requests; change issues, configuration, README, benchmark results or historical experiment results.

All new code, experiments, documentation, tests, branches and commits are created in `samehelmalt/LitBioRAG-v2-Production` only.

```text
samehelmalt/LitBioRAG  ──inspect / learn──▶  LitBioRAG-v2-Production
   (thesis, read only)                       (production, read + write)
```

The direction is never reversed. Production code may originate from thesis code; thesis code is never modified by production development.

## Verification performed before the first commit (2026-10-04)

1. `samehelmalt/LitBioRAG-v2-Production` exists and accepts pushes (confirmed via the session's repository listing).
2. Working clone remote: `https://github.com/samehelmalt/litbiorag-v2-production`, branch `main`.
3. The thesis clone's push URL was set to `DISABLED` locally so an accidental push from it fails.
4. No branch, commit or push was made to `samehelmalt/LitBioRAG`.

If there is ever uncertainty about which repository is the write target: stop, do not write, ask.

## What is deliberately not copied from the source

- Credentials, tokens, API keys (the thesis reads `OPENAI_API_KEY` from the environment and a UI field; v2 reads it from the environment only).
- Local machine and Google Drive paths (`/content/drive/MyDrive/...`).
- Model weights, FAISS indexes, corpus files (never in git in either repository).
- The thesis frontends, the Split-Brain Ensemble, the two-stage reranker, the SLSR synonym dictionary and the LitRag generator (see `docs/PRODUCTION_AUDIT.md` for the reasons).
