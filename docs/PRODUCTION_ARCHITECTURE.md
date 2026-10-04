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

## Pending sections

Generation contract (Step 6) · Claim verification (Step 7) · Gate v2 (Step 8) · API, observability, security (Step 9) · Frontend (Step 10) · Performance (Step 11).
