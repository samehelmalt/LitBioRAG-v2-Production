# Production Audit

Audit of the thesis system (`samehelmalt/LitBioRAG`, engine v5.5.21) performed before any production behaviour was changed. Sources: thesis v10 (18 June 2026), `src/engine_v5521.py` (line numbers below), `src/biorageval.py`, `kedarag/`, both frontends, and the 18 master-run CSVs (`Results/V5521_Run/`, 4,752 rows). Every statistic marked ★ was recomputed from those CSVs for this audit; the CSVs themselves were not touched. Architecture facts are summarised in `docs/THESIS_BASELINE.md`.

Decisions received from the owner on 2026-10-04 are recorded in §19.

---

## 1. Current thesis architecture

Eleven-step ladder S-1 → S7 (NoRAG, dense, +BM25, +RRF, +cross-encoder, +two-stage cascade, +SLSR, +DR-RAG routing, +DR-RAG decomposition, +Split-Brain Ensemble, +Six-Signal Gate). One 8,046-line module holds configuration, retrieval, generation, four evaluators, the gate, the benchmark loop and status rendering. Two Gradio frontends (1,795 and 1,595 lines) hold citation/NCBI logic. A `kedarag/` package and `biorageval.py` library duplicate parts of the evaluation. Models and indexes live on Google Drive; paths are hard-coded (`/content/drive/MyDrive`, L86–124).

## 2. Current retrieval architecture

Dense (FAISS HNSW SQ8 per year, document-level) + BM25 (`rank_bm25` in memory, `lower().split()` tokens, median-score cut) → dynamic RRF (base 0.60 BM25 / 0.40 dense, heuristic adjustments, k = 60) → reranking over `max(100, 12k)` candidates.

Two reranking modes:
- S3a one-stage: `MedCPT-Cross-Encoder` over the whole pool. R@K ★ 0.660.
- S3 two-stage (**used by the final system S4–S7**): stage 1 is a "bi-encoder reranker" bound to the same dense model that produced the dense half of the pool (`BiomedicalBiEncoderReranker.bind(embedder)`, L2231); it keeps `max(4k, 30)`; stage 2 is the cross-encoder. R@K ★ 0.583.

Both rerankers truncate documents to 512 **characters** (`[:512]`, L2247, L2341), roughly the first 80 words of an abstract; MedCPT-CE accepts 512 tokens.

## 3. Current generator architecture

Six 1.5–3B generators from 2024 (Qwen2.5-3B, Qwen2-1.5B, DeepSeek-R1-Distill-Qwen-1.5B NF4, DeepSeek-R1-Distill-Llama-3B community, Llama-3.2-3B, LitRag LoRA). Per-model prompt templates; FACTOID/YESNO forced to "Line 1: YES or NO"; when evidence is retrieved the prompt appends "MUST NOT abstain" (L1736, L1765, SBE prompt L1975). Whole-abstract context at 12,000 chars. Optional SBE: generator writes bulleted claims, a fixed DeepSeek-1.5B head writes the answer, the two are fused by NLI attribution coverage. Yes/no SBE prompt tells the model to pick the majority side when documents conflict.

## 4. Current verification architecture

`DeBERTa-v3-base-mnli-fever-anli` (184M) as the only NLI model; used for claim verification, SBE head selection, DR-RAG sub-answer verification, the gate's NLI and contradiction signals, and the NLI evaluation column. Premise chunked at 400 tokens; the chunk with the most confident *label* wins, so a confident NEUTRAL beats a less confident ENTAILMENT (L4089). Claims are regex sentence splits. Numeric check = number-string containment, 1.0 when the answer has no numbers (L4259). Entity check = regex query terms found in the context, not in the answer (L4264). Citations: sentences without `[Doc N]` are deleted; no check that the cited document supports the sentence (L1055). EAE adds cosine-similarity "support" at ≥ 0.72 with the retrieval encoder.

## 5. Current SSEG architecture

Linear composite 0.30 NLI + 0.20 numeric + 0.10 entity + 0.25 retrieval-confidence + 0.10 (1−contradiction) + 0.05 (1−unsupported); PASS ≥ 0.28 with contradiction ≤ 0.25; WARN ≥ 0.18; unsupported > 0.60/0.85 forces WARN/BLOCK; two BLOCK→WARN rescues (high retrieval confidence; gold-answer similarity, the latter inactive by an attribute-name mismatch, L4703). Benchmark post-gate score: `0.90 if BLOCK else faithfulness` (L6820).

## 6. Retrieval failure modes

★ From the 18 CSVs (medcpt × qwen_2.5_3b, representative):

| Failure | Evidence | Lines |
|---|---|---|
| Cascade discards BM25-only documents | S2 0.618 → S3 0.583; S3a 0.660. Final system inherits S3. | L2363–2420 |
| Reranker input truncated to 512 chars | both rerankers | L2247, L2341 |
| Toy BM25: no stemming/punctuation handling, median cut, 18 s median latency | S1 latency 18.3 s vs S0 6.1 s | L2196–2226 |
| Dense-only had the gold doc, pipeline lost it | Trifluridine/Tipiracil S0 0.667 → S3a 0; m6A 0.25 → 0; ovarian reclassification 1.0 (S0–S2) → 0 (S3, S3a) | — |
| Zero recall at S3a on 7 of 24 queries | ADAM17 in NK cells; rotator cuff treatments; insulin-resistance risk factors; C. difficile aerobe; m6A; Trifluridine; ovarian reclassification | — |
| Query understanding by regex; 25-entry synonym dictionary; " review"/" meta-analysis" suffix as the common expansion | SLSR S3→S4 delta +0.000 | L568, L813, L2649 |
| RRF weights heuristic, never tuned; "dense confidence" input is the spread statistic that does not discriminate | see §9 | L2461 |
| Document-level index only; no passage retrieval | evidence selection is regex sentence splitting of whole abstracts | L2131 |
| DR-RAG decomposition replaces the evidence set | S4→S5 R@K −0.056, +31 s latency | L3580 |

## 7. Generator failure modes

★ Yes/no accuracy at S3a: Llama-3B 0.76, DeepSeek-3B 0.71, DeepSeek-1.5B 0.52, Qwen-1.5B 0.24, Qwen2.5-3B 0.24, LitRag 0.10. NoRAG (S-1): Qwen2.5-3B 0.29 — retrieval *lowers* the structured models' accuracy. At S7 all generators 0.33–0.48; SBE (S5→S6) drops NLI faithfulness 0.273 → 0.138 and Llama-3B yes/no 0.76 → 0.33.

Named modes: forced answering ("MUST NOT abstain"); majority-vote instruction on conflicting evidence; chain-of-thought heads emitting unverifiable reasoning; citation tokens without support; polarity errors on negated findings (the Hypothesis Reconstructor exists because of this); numeric claims unverified beyond substring match; context truncation at 12,000 chars with whole abstracts.

## 8. Verification weaknesses

1. One model family for verification, selection and evaluation (circularity named by the thesis §5.5).
2. Base-size general-domain NLI; thesis §1.2 itself calls it miscalibrated for paraphrase.
3. Most-confident-label chunk aggregation (L4089).
4. Regex claim splitting; no evidence spans; no citation-to-claim binding.
5. Numeric and entity checks are near-noise (§9).
6. EAE "support" is topical cosine similarity, not entailment.
7. Abstention text gets faithfulness 0.90 by regex (L4197).

## 9. False-PASS risks

★ Measured on the 432 S7 rows:

- 64 PASS rows. Among the 25 yes/no PASS rows, **14 (56%) have the wrong answer**.
- **22 of 64 PASS rows (34%) have Recall_K_Norm = 0**: the gold document was never retrieved.
- Mean unsupported-claim rate within PASS rows: 0.44. Mean Signal_Entity 0.20; mean Signal_Retrieval 0.238 (BLOCK rows: about the same, 0.236 overall).
- Signal_Numeric = 1.0 on 69.9% of rows because the answer contains no numbers.
- Conversely 21 of 66 blocked yes/no answers (32%) were correct and 59% of BLOCK rows had the gold document in the top 5.

Structural causes:
- **Gold label inside the gate.** In the benchmark loop `HypothesisReconstructor.reconstruct(query, answer, yn_label, q_type)` takes the hypothesis direction from the gold label (L2920–2924, called at L6739), then `faithfulness = 0.50·recon + 0.30·faith + 0.20·yn_correct` (L6757) feeds `Signal_NLI`. Half of the S7 rows are yes/no. The gate's measured behaviour on them is not reproducible without gold labels.
- **Constant post-gate score.** All 221 BLOCK rows have Post_Gate_Faith = 0.900. 0.512 × (0.90 − 0.016) = 0.452: the headline effect equals the BLOCK rate times a constant.
- Linear composite: 0.20 for "no numbers" + 0.25 near-noise retrieval confidence can carry an answer with 0.44 unsupported claims across the 0.28 PASS line.
- Rescue rules lower severity; one reads gold-answer similarity by design.

## 10. Performance bottlenecks

★ Median latency per step (s): S0 6.1 · S1 18.3 · S2 18.1 · S3 18.2 · S3a 34.2 · S4 33.6 · S5v1 50.7 · S5 65.4 · S6 64.8 · S7 64.0; S7 mean 75.1, max 243.8 (Colab L4).

- BM25 in pure Python over 1.78M docs: +12 s.
- Cross-encoder over 100 pairs at batch 8 (L2321): +16 s.
- DR-RAG decomposition (3 extra retrievals + 3 sub-generations + NLI): +31 s.
- Models loaded/unloaded through a global `ModelCache` with manual eviction; no batching across requests; no streaming.

## 11. Security weaknesses

- Prompt text from retrieved documents is concatenated into the prompt unfiltered; no injection detection.
- `OPENAI_API_KEY` accepted from a Gradio text box and written to `os.environ` (L4281).
- `_bootstrap()` runs `pip install` at import time (L16–18) and mounts Google Drive.
- NCBI calls with no API key, no rate limiting, no timeout policy beyond 10–15 s, results not cached.
- Every component swallows `Exception` and degrades silently (reranker disabled → `docs[:top_k]`; NLI unavailable → sub-answer kept). For a safety layer, silent degradation is a failure mode.
- Business logic in the frontend; no authentication or request limits on the API surface.
- No secrets found in the repository.

## 12. Proposed production architecture

```text
Query → Query understanding (NER, acronyms, MeSH, intent; original query preserved)
      → Retrieval: BM25 (Lucene, on disk) ‖ Dense (FAISS IVF-PQ/DiskANN, on disk) ‖ entity/metadata filters
      → RRF (benchmarked k, weights)
      → Single-stage cross-encoder over the full pool at full token length
        (a cascade only with a different, cheaper stage-1 model and a measured gain)
      → Passage-level evidence selection with provenance (PMID/DOI, title, year, source_type, span)
      → Generator (typed output: claims, citations, abstain flag, conflict flag; abstain-first prompt)
      → Verification per claim: evidence map → entailment/contradiction (verifier ≠ generator family)
        → entity consistency → numeric consistency (unit-aware) → citation resolution
      → Gate v2: ordered hard checks → calibrated confidence → PASS / WARN / BLOCK
      → API response with evidence, verdicts, uncertainty, conflicts
```

Service layer: FastAPI (`app/api`), typed contracts (`app/core`), pydantic-settings config, structured logs (`app/observability`), model registry (`models/registry`). Offline evaluation in `evaluation/` with tuning/validation/calibration/test/adversarial splits.

## 13. Files to reuse (port after review, with tests)

| Thesis location | What | Destination |
|---|---|---|
| `rrf_fusion` L2514 | RRF core (without the confidence reweighting wrapper) | `app/retrieval/fusion.py` |
| `FaissIndex` L2131 | HNSW load/search with per-call efSearch pattern | `app/retrieval/dense.py` (adapted to on-disk IVF-PQ) |
| `ExtractedGTLoader` L1087, `BenchmarkDatasetLoader` L4893 | BioASQ/PubMedQA loading with corpus-residency check | `evaluation/datasets/` |
| `DualEvaluator._compute_ranking_metrics` L4474 | R@k, MRR, nDCG, corpus-normalized variants | `evaluation/metrics/retrieval.py` |
| `HypothesisReconstructor` L2894 | declarative yes/no hypotheses — **reimplemented from the answer's polarity only** | `app/verification/hypothesis.py` |
| `EpistemicGate` L4665 | PASS/WARN/BLOCK vocabulary, contradiction ceiling, unsupported overrides, per-signal explanation string | `app/safety/gate.py` (structure only) |
| `frontend_v5521.py` L527–860 | NCBI esummary/efetch calls | `app/api/sources.py` (with API key, cache, rate limit) |
| `Results/V5521_Run/*.csv`, `Results/calib/` | thesis results, read-only copy | `evaluation/benchmarks/thesis_baseline/` |
| `benchmark_data/extracted queries/*.json` | the 24 + 200 query files | `evaluation/datasets/thesis/` |

## 14. Files to rewrite

Retrieval pipeline (`retrieve`, BM25, rerankers, RRF weighting); query understanding (`classify_query_mode`, `extract_query_entities`, `slsr_expand`); generation prompts and `GeneratorPipeline`; `ContextPacker` → passage-level evidence selection; `ClaimLevelEvaluator`, `ClaimLevelNLI`, `NLIHelper` → claim verification pipeline; `CitationEnforcer` → citation resolution; `EpistemicGate` decision logic; `SystemConfig` → pydantic-settings; `ModelCache` → model registry; logging; both frontends → thin client.

## 15. Files not to copy

`TwoStageCascadeReranker` (self-bi-encoder stage), `SplitBrainEnsemble`, `SelfConsistencyVoter`, `DrRagDecomposerV2`/`DrRagDecomposer` as implemented, `slsr_expand` dictionaries, `_bootstrap()` and Drive mounting, `Qwen3Router` (re-evaluate later), `kedarag/` TF-IDF/ROUGE wrappers, `RealRAGASEvaluator` runtime wiring, `HighConfRescue`/`SemRescue`, `post_gate_faithfulness = 0.90`, abstention-regex faithfulness 0.90, LitRag weights/config, DeepSeek-R1-Distill-Llama-3B and Qwen2-1.5B entries, `GeneratorProfile` regex extractors, the `MUST NOT abstain` prompt fragments, `ThesisBenchmark` loop (replaced by the harness).

## 16. Baseline experiments

1. **Harness check**: reimport the 18 thesis CSVs and reproduce the ladder table in `THESIS_BASELINE.md` §10 to 3 decimals.
2. **Production baseline (Step 4)**: thesis algorithms reimplemented cleanly — BM25 + dense + RRF + single-stage CE at full length — on the harness; report R@1/5/10/20, MRR, nDCG, first relevant rank, by query type; p50/p95/p99, VRAM, RAM.
3. **Reranker ablation**: one-stage vs. two-stage-with-different-model; 512 chars vs. full tokens.
4. **BM25 ablation**: `lower().split()` vs. biomedical analyzer; median cut on/off.
5. **Gate re-scoring without gold labels**: recompute Signal_NLI on the 432 S7 rows using answer polarity only; report how many PASS/WARN/BLOCK decisions change. This quantifies §9's leakage.
6. **False-PASS proxies on the thesis rows**: yes/no wrong ∧ PASS; gold-doc-missing ∧ PASS; define the proxy labels used until human labels exist.

## 17. Recommended model candidates (to benchmark, one winner per role)

Hardware constraint: GPUs with 12 GB VRAM, 16 GB system RAM, quantized models (§19). Candidates are sized accordingly.

| Role | Candidates | Selection metric |
|---|---|---|
| Dense encoder | MedCPT (baseline), Qwen3-Embedding-0.6B, BGE-M3, BMRetriever-410M | recall@100 of the candidate pool, encode throughput |
| Lexical | Lucene/Pyserini BM25 with biomedical analyzer; OpenSearch | recall@100, latency |
| Reranker | MedCPT-CE (baseline), BGE-reranker-v2-m3, Qwen3-Reranker-0.6B; optional 4-bit Qwen3-Reranker-4B | R@5/10, nDCG, ms per 100 pairs |
| Query understanding | scispaCy `en_core_sci_md` + abbreviation detector; UMLS/MeSH lookup; Qwen3-1.7B for intent only | entity F1 on a labelled sample; retrieval effect of conditional expansion |
| Generator (≤ 12 GB at 4-bit) | Qwen3-8B, Qwen3-14B (4-bit ≈ 9 GB), Gemma-3-12B-it, MedGemma-4B-it, Llama-3.1-8B-Instruct | grounding, abstention compliance, citation validity, yes/no accuracy, VRAM, tokens/s, quantization loss vs. bf16 |
| Generator (hosted, optional) | 27B-class or API model behind the same contract | same, plus cost and latency |
| Claim verifier | Bespoke-MiniCheck-7B (4-bit) or MiniCheck-Flan-T5-L; DeBERTa-v3-large-mnli-fever-anli-ling-wanli; DeBERTa-v3-base (baseline) | precision/recall on proxy-labelled claims; must differ from generator family |
| Entity / numeric | scispaCy NER + UMLS linker; unit-aware numeric matcher with tolerance | agreement on adversarial swaps |
| Offline judge | strong LLM-as-judge for evaluation only, never at runtime | — |

## 18. Hardware requirements

Target: GPUs with 12 GB VRAM, 16 GB system RAM. Consequences:

- A 27B generator does not fit: 4-bit weights alone are ~15–16 GB. 27B is admissible only as a hosted/API candidate or a one-off CPU-offload measurement expected to fail the latency budget.
- Full current PubMed (~37M abstracts → ~80–100M passages) cannot be served from RAM. BM25 on disk (Lucene), dense via FAISS IVF-PQ or DiskANN with memory-mapped vectors, metadata in SQLite/DuckDB keyed by PMID/DOI. Embedding the full corpus with a 0.6B encoder on one 12 GB GPU takes days; Step 3 begins with a recent window (2015–present) plus bioRxiv/medRxiv and backfills the full baseline as a resumable job.
- Per-request co-residency: reranker (0.6B, ~1.5 GB) + verifier (≤ 1B NLI or 4-bit 7B ≈ 5 GB) + 4-bit generator ≤ 14B (≈ 9 GB) exceed 12 GB together; load/unload in sequence or split across GPUs; Step 11 measures both arrangements.
- Disk: ~60–120 GB for indexes plus the metadata store at full-corpus scale.

## 19. Production risks and decisions

Decisions received 2026-10-04: corpus = full current PubMed + bioRxiv/medRxiv (preprints flagged `source_type=preprint` and placed below peer-reviewed sources in the evidence hierarchy); hardware as in §18; proxy labels only for now (yes/no correctness, gold-PMID presence), labelling export/import tooling to be built; false-PASS target set after Step 8 from the measured coverage–risk curve.

Risks:
1. **Proxy labels mis-estimate false PASS.** Yes/no correctness and gold-PMID presence are necessary, not sufficient, conditions for a faithful answer. Documented in `TRUST_AND_SAFETY.md` until human labels exist.
2. **Hardware forces quantization** whose quality cost must be measured per role; a 4-bit verifier that loses recall raises false PASS directly.
3. **Full-corpus indexing time and disk** on the stated hardware; mitigated by windowed start and resumable backfill.
4. **Preprint handling**: non-peer-reviewed sources must never be presented as equivalent to journal articles; evidence hierarchy and UI labelling are part of Step 7–10 acceptance.
5. **Benchmark distribution**: BioASQ/PubMedQA questions are not production questions; the held-out sets must include free-form literature questions.
6. **No improvement is claimed until measured.** `BASELINE_RESULTS.md` carries "NOT YET MEASURED" until Step 4 runs.
