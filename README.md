# LitBioRAG v2 Production

Evidence-gated biomedical literature question answering.

LitBioRAG v2 is the production successor of the MSc thesis system [samehelmalt/LitBioRAG](https://github.com/samehelmalt/LitBioRAG). It retrieves from PubMed and biomedical preprints, generates answers only from retrieved evidence, verifies every important claim against that evidence, and routes each answer to **PASS / WARN / BLOCK**. Abstaining is a valid outcome.

> Not a clinical tool. Nothing produced by this system is medical advice.

## Success criterion

The system is judged by one statement:

> It generates answers only when the evidence justifies them, provides verifiable provenance for important claims, detects unsupported or contradictory claims, and abstains safely when evidence is insufficient.

The primary safety metric is the **false-PASS rate**: the share of released answers that contain unsupported, contradicted or mis-cited claims. Coverage is secondary to it.

## Relationship to the thesis

The thesis repository is a read-only reference; see [`docs/REPOSITORY_BOUNDARY.md`](docs/REPOSITORY_BOUNDARY.md). Nothing is copied without review. The audit of what to reuse, rewrite and discard is in [`docs/PRODUCTION_AUDIT.md`](docs/PRODUCTION_AUDIT.md); the thesis architecture and historical numbers are recorded in [`docs/THESIS_BASELINE.md`](docs/THESIS_BASELINE.md).

## Layout

```text
app/            service code: api, core, retrieval, reranking, evidence, generation,
                verification, routing, evaluation, safety, observability
models/         model registry and adapters (no weights)
evaluation/     offline harness: datasets, benchmarks, diagnostics, calibration, adversarial
tests/          unit, retrieval, generation, verification, integration, regression, security
configs/        YAML configs (no paths, no secrets)
scripts/        corpus, index and benchmark scripts
deployment/     docker and CI
docs/           architecture, audit, evaluation, trust and safety, API, deployment
data/           local corpus and indexes (git-ignored)
frontend/       thin client over the API
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # fill in locally, never commit
ruff check . && pytest
```

Heavy ML dependencies are optional extras (`pip install -e ".[ml]"`) so the test suite and documentation build on any machine.

## Status

Phase 0 (repository bootstrap and audit). No production behaviour has been measured yet; `docs/BASELINE_RESULTS.md` says so explicitly. No improvement over the thesis is claimed until the evaluation harness in `evaluation/` reports it.
