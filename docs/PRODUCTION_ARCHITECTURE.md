# Production Architecture

Living document. Sections are added as the component they describe exists in the repository and has tests; design intent for later components lives in `docs/PRODUCTION_AUDIT.md` §12.

## Principles

1. The evidence is authoritative; the generator is not.
2. Every unit of evidence carries provenance (PMID/DOI, title, year, source type, passage id, character offsets, score). Nothing without provenance reaches the generator, the verifier or the user.
3. Components are swappable behind small protocols (`Encoder`, `Reranker`, later `Generator`, `Verifier`) so the evaluation harness can benchmark one role at a time.
4. Offline evaluation (`evaluation/`) and runtime (`app/`) share data models and nothing else; gold labels never enter runtime code.
5. Fail loud: a missing index or model raises; it does not silently degrade to a weaker path.

## Corpus and indexes (Step 3)

See `docs/CORPUS.md` and `docs/DEPLOYMENT.md`. A bundle = `metadata.sqlite` (documents, passages) + `bm25/` (Tantivy) + `dense/` (FAISS) + `manifest.json`. Built on a build host, verified by sha256 on the serving host, loaded read-only / memory-mapped.

## Retrieval (Step 4)

```text
query ──► BM25 (Tantivy; stemmed + normalized views)  ─┐
      └─► dense (FAISS; encoder fixed by the bundle)   ─┼─► RRF (k, weights from configs/retrieval.yaml)
                                                        │
                                                        ▼
                                  single-stage cross-encoder over the top `rerank_pool` passages
                                  (token-level truncation at the model max; query never cut)
                                                        │
                                                        ▼
                                  passages → documents (max score; best `passages_per_doc` kept)
                                                        │
                                                        ▼
                                  evidence selector → EvidenceUnit[] (provenance-complete, preprints flagged)
```

Modules: `app/retrieval/bm25.py`, `dense.py`, `fusion.py`, `pipeline.py` (`HybridRetriever`, modes `bm25 | dense | hybrid | hybrid+rerank`), `app/reranking/base.py` + `cross_encoder.py`, `app/evidence/selector.py`.

Decisions carried from the audit:
- **No retriever-as-reranker cascade.** The thesis pre-filtered the pool with its own dense encoder and lost BM25-only documents (S3 0.583 vs S3a 0.660). The production reranker scores the whole pool in one stage. A cascade is admissible only with a different, cheaper stage-1 model and a measured gain.
- **Passage-level units** with offsets instead of whole abstracts cut at 512 characters.
- **Audit on every result**: rank of each candidate document at each stage, per-stage timings. The runner turns this into failure codes.
- **Static RRF weights** from config; the thesis's heuristic re-weighting is not ported because its confidence input did not discriminate.

Not yet present and deliberately so: query expansion (Step 5e, conditional and benchmarked), entity/metadata filters (Step 5e), routing (deferred).

## Evaluation harness (Step 2)

`evaluation/`: loaders, split manifest (tuning / validation / calibration / test / adversarial, no text or PMID overlap), metrics (retrieval, claims, trust, performance), proxy labels, thesis import with a regression test, runners (`evaluation/runners/retrieval.py`). See `docs/EVALUATION.md`.

## Model registry (Step 5-prep)

`configs/models.yaml` → profile → role → `ModelSpec`; adapters keyed by `kind` (`models/registry/`). Swapping a model is a YAML or `LITBIORAG_MODEL__<ROLE>` change. The loader enforces that generator, verifier and offline judge come from different model families, and hands out calibrated thresholds only for the model id they were fitted on. See `models/registry/README.md`, `docs/MODEL_SELECTION.md`.

## Generation (Step 6)

```text
EvidenceUnit[] ──► prompt answer-json-v1 (abstain-first; never-invent list; conflicts reported, not voted;
                   passage text is data) ──► Generator adapter ──► contract parser ──► GeneratedAnswer
```

`GeneratedAnswer` = answer text + atomic `Claim`s (citation ids copied from the evidence, critical flag, optional proposition fields) + abstention flag/reason + conflicts. Unparseable output is a forced abstention. Cited ids not in the evidence set are recorded as unknown. No evidence → abstention without a model call. Adapters: `hf_causal`, `openai_compatible`, `anthropic`, `template` (CI). Modules: `app/generation/contract.py`, `prompts.py`, `base.py`, `service.py`.

## Claim verification (Step 7)

```text
GeneratedAnswer ──► prepare_claims ──► per claim:
   resolve citations (fabricated → fail)
   score (passage, claim) pairs with the Verifier (cited passages first, then best lexical)
   label: contradicted if max contradiction ≥ τ_c, else supported if max entailment ≥ τ_e, else unsupported
   entity consistency vs supporting passages · numeric consistency (unit class, tolerance)
   conflict record when one source supports and another contradicts
──► VerificationReport (separate fields; policy trace)
```

Modules: `app/core/verdicts.py` (shared with `evaluation/metrics/claims.py`), `app/verification/{base,claims,evidence_map,citations,entities,numeric,conflict,pipeline}.py`. Verifier adapters: `hf_seq_cls` (token-level chunking; max entailment **and** max contradiction across chunks), `overlap` (CI). The verifier must be a different model family from the generator (registry rule). Calibrated thresholds from the registry override `configs/verification.yaml`. What each check catches and cannot catch: `docs/TRUST_AND_SAFETY.md`.

End-to-end without weights: `python -m scripts.demo_pipeline --bundle <bundle> --profile test --question "..."` (covered by `tests/integration`).

## Pending sections

Gate v2 and calibration (Step 8) · PPTX content adapter (Step 3e) · API, auth, entitlements, observability, security (Step 9) · Web UI and trust page (Step 10) · Model benchmarks (Step 5, needs the build host) · Performance (Step 11).
