# AEOS Medical — evidence-gated biomedical question answering

AEOS Medical (repository: LitBioRAG v2 Production) answers biomedical questions from retrieved
evidence only, verifies every claim against the cited passage, decides whether to deliver the
answer (**ANSWER / ANSWER WITH CAUTION / CLARIFY / ABSTAIN / ESCALATE**), and sends withheld
answers to a human reviewer. Users can search the indexed corpus (PubMed, bioRxiv/medRxiv), their
own uploaded files (PDF, Word, Excel, PowerPoint, text), or both.

> Not a clinical tool. Nothing produced by this system is medical advice.

It is the production successor of the MSc thesis system
[samehelmalt/LitBioRAG](https://github.com/samehelmalt/LitBioRAG) (read-only reference; see
`docs/REPOSITORY_BOUNDARY.md`). The thesis audit is in `docs/PRODUCTION_AUDIT.md`.

## Status in one paragraph
Architecture and product are code-complete and tested (CI runs lint, 220+ tests and the adversarial
substitution suite). **No accuracy number has been measured yet**: every benchmark needs a real
model profile (an API key for the hosted generator, or a GPU host) and a real corpus bundle. Until
then the trust page says NOT YET MEASURED and the gate runs with heuristic, uncalibrated defaults.
`docs/STATUS.md` lists exactly what is done and what is missing.

## Quick start (serving host)
```bash
pip install -e ".[dev,eval,index,api,kb,pptx]"      # add ml for local GPU models
cp .env.example .env                                 # fill LITBIORAG_JWT_SECRET, OPENAI_API_KEY …
python -m scripts.build_indexes --docs docs.jsonl --out data/artifacts/demo --corpus-id demo \
    --encoder hash:128                                # tiny demo bundle; real bundles: docs/CORPUS.md
python -m scripts.serve --bundle data/artifacts/demo --profile test --port 8471
# open http://localhost:8471  (UI)   http://localhost:8471/api/docs  (OpenAPI)
```
Many concurrent users: `compose.gpu-vllm.yaml` (vLLM generator + Redis on one GPU host) or
`compose.scale.yaml` (API replicas behind nginx); see docs/DEPLOYMENT.md "Scaling".

Docker (local test, no preparation): `docker compose up --build`, then open
http://localhost:8471 . The first start builds a demo bundle from `scripts/demo/demo_docs.jsonl`
with deterministic test doubles and prints three demo tokens (user, reviewer, admin) to paste into
"Sign in". For real models set `PROFILE=hosted` and `OPENAI_API_KEY` in `.env` and point
`CORPUS_ID` at a fetched bundle under `./data/artifacts`. Port 8471 is used on purpose to avoid
the usual 8000/8080/3000.

Tokens: the UI accepts a JWT (HS256 by default, claims `sub`, `roles`, `courses`, `aud=litbiorag`)
signed with `LITBIORAG_JWT_SECRET`, or an API key issued on the API Access page.

## Layout
```text
app/        api (FastAPI), service (one pipeline), retrieval, reranking, evidence, generation,
            verification, safety (gate), review (human queue), kb (uploads), corpus (index
            management), web (UI)
models/     registry and adapters (no weights)
evaluation/ offline harness: datasets, splits, metrics, adversarial suites, runners
configs/    models, retrieval, generation, verification, gate, entities, resources, api, review,
            corpus, pricing (no secrets)
scripts/    corpus builders, index build/update, artifacts, baselines, calibration, serve, release
tests/      unit, retrieval, generation, verification, safety, security, api, kb, review, corpus
docs/       audit, architecture, evaluation, trust & safety, API, security, governance, status
```

## Development
```bash
set -o pipefail && ruff check . && python3 -m pytest -q
python -m scripts.demo_pipeline --bundle data/artifacts/demo --profile test --question "…"
python -m scripts.run_adversarial --bundle data/artifacts/demo --profile test --out runs/adv
python -m scripts.release --version 0.9.0        # checks + dist/aeos-medical-0.9.0.zip
```
Change models in `configs/models.yaml` or with `LITBIORAG_MODEL__<ROLE>=kind:model`; never in code.
