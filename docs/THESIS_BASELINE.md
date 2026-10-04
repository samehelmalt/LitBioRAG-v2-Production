# Thesis Baseline (LitBioRAG v5.5.21)

This document records what the MSc thesis system is and what it measured, so that production work can be compared against it. It is descriptive. Nothing here has been re-run; every number is quoted from the thesis master-run CSVs in `samehelmalt/LitBioRAG` (`Results/V5521_Run/thesis_ladder_*.csv`, 18 files, 4,752 rows, executed 12–13 June 2026) or from the thesis text (v10, 18 June 2026). Where this document recomputes an aggregate from the CSVs, it says so. The original results are not modified.

Source files referenced: `src/engine_v5521.py` (8,046 lines; line numbers below refer to it), `src/biorageval.py`, `kedarag/`, `src/frontend_v5521.py`, `src/Frontend_lite_prod_v5521.py`.

---

## 1. Purpose and claims of the thesis

Research questions: RQ1 retrieval competitiveness and generation lift; RQ2 interactive latency (< 90 s median on one L4); RQ3 selective prediction via an epistemic gate; RQ4 EAE vs NLI faithfulness disagreement.

Confirmatory hypotheses: H1 R@K at S3a ≥ 0.567; H2 pre→post gate delta ≥ +0.40 (p < 0.001, d > 0.8); H3 EAE exceeds NLI by ≥ +0.25 on average. E1 (exploratory): the Split-Brain Ensemble has generator-dependent effects.

Headline result as stated: abstention-adjusted post-policy score gain of +0.452 (paired t(431) = 21.20, d = 1.02, n = 432 S7 rows).

## 2. Corpus and indexes

- 1,784,129 unique PMIDs, PubMed/PMC, publication years 2015–2016, harvested via E-utilities (thesis §3.7.1).
- Dense: FAISS HNSW with SQ8, one index per year, `efSearch = max(2·k, 64)` set per call (`FaissIndex.search`, L2158).
- Lexical: `rank_bm25.BM25Okapi` built in memory per year shard; tokenization is `text.lower().split()`; a median-score threshold filter is applied to the top 3k candidates (`BM25ShardIndex`, L2196–2226).
- Document-level units (one vector and one BM25 document per abstract). No passage index.
- Corpus pre-verification excludes benchmark queries whose gold PMIDs fall outside 2015–2016.

## 3. Retrieval architecture

Pipeline (`LitBioRAGEngine.retrieve`, L5371–5475):

1. Optional SLSR expansion (not for FACTOID) → up to 5 query variants.
2. Dense top-`max(6k, 50)` and BM25 top-`max(6k, 50)` per variant; de-duplicated by PMID.
3. Dynamic RRF (`compute_rrf_weights`, L2461): base BM25 0.60 / dense 0.40, adjusted by query length, quotes, a dense "confidence" spread statistic and the BM25 top score; `rrf_k = 60`.
4. Reranking over `max(100, 12·k)` candidates via `TwoStageCascadeReranker` (L2363):
   - **one-stage (S3a)**: cross-encoder over the full pool;
   - **two-stage (S3, default for S4–S7)**: stage 1 = `BiomedicalBiEncoderReranker`, which is bound to the *same* dense encoder used for retrieval (L2231) and keeps `max(4k, 30)`; stage 2 = cross-encoder.
5. Both rerankers truncate each document to 512 **characters** (`[:512]`, L2247, L2341).
6. Retrieval confidence for the gate = entropy-based statistic over the final top-k scores (`_compute_retrieval_confidence_entropy`, L720).

Models: dense encoders MedCPT-Query-Encoder (`ncbi/MedCPT-Query-Encoder`), author-fine-tuned PubMedBERT ("ft_pubmedbert"), author-fine-tuned MiniLM ("ft_minilm_abstract"); cross-encoder `ncbi/MedCPT-Cross-Encoder` when cached, else `cross-encoder/ms-marco-MiniLM-L-6-v2` (L370–399).

## 4. Query understanding and expansion

- `classify_query_mode` (L2622): yes/no if the query starts with one of 15 polar verbs, else keyword tables → FACTOID / COMPARATIVE / NORMATIVE / CAUSAL / PROCEDURAL, default FACTOID. A learned router (`Qwen3Router`, Qwen3-0.8B, L2685) exists but was disabled in the master run.
- `extract_query_entities` (L568): regex.
- SLSR (`slsr_expand`, L2649): 25-entry hand-written synonym dictionary (`_BIOMEDICAL_SYNONYMS`, L813), a 6-entry "historical" trigger table, and a fallback that appends " review", " randomized trial", " meta-analysis". Suppressed for FACTOID.

## 5. DR-RAG (NC-1)

- V1 (S5v1): route COMPARATIVE/CAUSAL/NORMATIVE to `Full_Stack`, others to `Hybrid_Rerank` (L6553).
- V2 (S5, `DrRagDecomposerV2`, L3580): decompose into 2–3 sub-questions with the generator (temperature 0.05, ≤100 new tokens); retrieve top-5 per sub-question; generate a sub-answer; verify each sub-answer by NLI against a premise built from the first 300 characters of 2 documents (L3719); keep if entailment ≥ 0.5; synthesize kept sub-answers into the context. YESNO never decomposed.

## 6. Generation

Six generators (L404–411): Qwen2.5-3B-Instruct, Qwen2-1.5B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B (NF4), DeepSeek-R1-Distill-Llama-3B (community distillation), Llama-3.2-3B-Instruct, LitRag (LoRA r=16 α=32 on Llama-3.2-3B, 50k rows from BioASQ-train + PubMedQA-L + synthesized pairs, ~26 h on one L4).

Prompting (`GeneratorPipeline._prompt_*`, L1734–1835): per-model templates. FACTOID/YESNO use "Line 1: YES or NO" strict format. When `evidence_available=True` the prompt appends "CRITICAL: Evidence IS present. Answer YES or NO. MUST NOT abstain." Temperature per mode (`MODE_TEMPERATURE`, L2608): FACTOID/YESNO 0.10, CAUSAL/COMPARATIVE 0.80, NORMATIVE 0.60; sampling disabled for FACTOID/YESNO.

Context (`ContextPacker`, L1545): whole abstracts as `[Doc i | PMID | Score] Title / Text` blocks, 12,000-char budget. `CitationEnforcer` (L1055) keeps sentences containing `[Doc N]`.

SBE (NC-3, `SplitBrainEnsemble`, L1893): claim head = generator under test produces ≤N bulleted claims; reasoning head = DeepSeek-R1-Distill-Qwen-1.5B produces the final answer; output = "Evidence summary: … Final answer: …". YESNO prompt instructs majority vote when documents conflict.

## 7. Verification and evaluation components

- `NLIHelper` (L3936): `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`; premise chunked at 400 tokens; per chunk the predicted label and its probability; the chunk with the highest label probability wins (L4089). Hypotheses matching abstention phrases return the sentinel `ABSTAIN`.
- `ClaimLevelEvaluator` (L4166): sentence-split answer → per-claim NLI against each `[Doc …]` block; faithfulness = supported / total, plus contradiction and unsupported rates. An abstaining answer is assigned faithfulness 0.90, contradiction 0.0, unsupported 0.10 (L4197–4200).
- `ClaimLevelNLI` (L3037): used for SBE outputs; per-claim max entailment over 400-word chunks; supported if ≥ 0.5.
- `HypothesisReconstructor` (L2894): for YESNO, rewrites "Does X improve Y?" + polarity into "X does (not) improve Y". In the benchmark loop the polarity direction is taken from the **gold** `yn_label` when present (L6739), and the rescored faithfulness is blended as `0.50·recon + 0.30·faith + 0.20·yn_correct` (L6757).
- Numeric fidelity: substring containment of number strings from the answer in the context; 1.0 if the answer has no numbers (L4259). Entity grounding: regex query entities found in context (L4264).
- EAE (`EvidenceAttributionEvaluator`, L3210): claim ↔ evidence-sentence cosine with the retrieval encoder; supported ≥ 0.72, partial ≥ 0.52; contradiction via the NLI model; type-specific blend (Appendix B of the thesis).
- KedaRag (`kedarag/`): TF-IDF/BioBERT/NLI/ROUGE wrapper; `biorageval.py`: standalone EAE library (defaults to `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` and `cross-encoder/nli-deberta-v3-xsmall`).
- RAGAS: `gpt-4o-mini` + `text-embedding-3-small` when `OPENAI_API_KEY` is set, on steps S3a/S5v1/S5/S6/S7 only; otherwise a lexical proxy.

## 8. Six-Signal Epistemic Gate (NC-4, `EpistemicGate`, L4665)

```text
composite = 0.30·NLI + 0.20·numeric + 0.10·entity + 0.25·retrieval_conf
          + 0.10·(1 − contradiction) + 0.05·(1 − unsupported)
PASS  if composite ≥ 0.28 and contradiction ≤ 0.25
WARN  if composite ≥ 0.18
BLOCK otherwise
override: unsupported > 0.85 → BLOCK; > 0.60 → WARN (if more severe)
rescues: retrieval_conf ≥ 0.80 → BLOCK becomes WARN;
         gold-answer similarity > 0.60 → BLOCK becomes WARN (inactive in the run: attribute name mismatch)
```

Held-out calibration (thesis §4.4.5, `Results/calib/`): 30 disjoint queries, one-stage CE configuration, re-derived thresholds PASS ≥ 0.42 / WARN ≥ 0.30.

Post-gate scoring in the benchmark: `post_gate_faithfulness = 0.90 if decision == "BLOCK" else faithfulness` (L6820; same rule at L5678 and L5789).

## 9. Benchmark protocol

- 24 queries (12 BioASQ Training 14B, 12 PubMedQA ori_pqal), all with gold PMIDs inside the 2015–2016 corpus. Types: 12 yesno, 6 factoid, 3 normative, 2 causal, 1 comparative. Full list: thesis Appendix E.
- 18 configurations = 3 encoders × 6 generators; 11 ladder steps each; 4,752 rows.
- Ladder: S-1 NoRAG · S0 dense · S1 BM25 · S2 +RRF · S3a +CE one-stage · S3 +two-stage cascade · S4 +SLSR · S5v1 DR-RAG routing · S5 DR-RAG decomposition · S6 +SBE · S7 +gate. **S4–S7 are built on the S3 (two-stage) path**, not S3a.
- `top_k = 5` for the reported recall; k = 10 variant reported in §4.3.4.

## 10. Historical results (quoted; aggregates recomputed from the 18 CSVs where marked ★)

| Step | R@K_norm ★ | MRR_norm ★ | NLI faithfulness ★ | median latency s ★ |
|---|---|---|---|---|
| S-1 | 0.000 | 0.000 | 0.472 | 6.5 |
| S0 | 0.472 | 0.434 | 0.233 | 6.1 |
| S1 | 0.604 | 0.505 | 0.234 | 18.3 |
| S2 | 0.618 | 0.559 | 0.264 | 18.1 |
| S3 | 0.583 | 0.522 | 0.280 | 18.2 |
| S3a | **0.660** | 0.566 | 0.285 | 34.2 |
| S4 | 0.583 | 0.522 | 0.289 | 33.6 |
| S5v1 | 0.583 | 0.522 | 0.288 | 50.7 |
| S5 | 0.528 | 0.440 | 0.273 | 65.4 |
| S6 | 0.583 | 0.522 | 0.138 | 64.8 |
| S7 | 0.583 | 0.522 | 0.138 (pre) / 0.590 (post) | 64.0 |

- S0 per encoder ★: MedCPT 0.399, ft-PubMedBERT 0.514, ft-MiniLM 0.503. S3a per encoder ★: all three 0.6597 (cross-encoder convergence).
- S7 gate distribution ★: PASS 64 (14.8%), WARN 147 (34.0%), BLOCK 221 (51.2%); pre-gate mean 0.138, post-gate mean 0.590, delta +0.452.
- S7 latency ★: median 64.0 s, mean 75.1 s, max 243.8 s (Colab L4).
- Held-out calibration (thesis Table 7): one-stage CE, thresholds 0.28/0.18 → +0.514; thresholds 0.42/0.30 → +0.530; coverage 41.7% → 38.7%.
- Weight sensitivity (`gate_sensitivity_report.txt`): ±0.05 per weight changes ≤ 1.2% of decisions.
- EAE vs NLI (thesis §4.5): EAE > NLI on 78.1% of 4,320 evaluable rows, mean delta +0.296.
- Error types across 4,752 rows (thesis Table 12): evaluation_ok 40.5%, retrieval_failure 34.2%, yn_wrong 10.9%, norag_baseline 9.1%, hallucination 3.5%, ranking_failure 1.2%, context_missing 0.6%.

## 11. Stated limitations (thesis §5.4–5.5)

Restricted 2015–2016 corpus; 24 unique questions; no human faithfulness labels; abstention-dependent interpretation of the gate effect; hand-specified weights and in-experiment threshold recalibration; six small generators, possible LitRag train/test overlap; single Colab L4 environment; the same NLI family used in verification, SBE selection, gate and evaluation; 51.2% BLOCK rate and 64 s latency not assessed with users.

## 12. Repository facts

- README lists `litbiorag_prod_chat.py`, `indexing/`, `requirements.txt`; none exist in the repository.
- No tests. No dependency manifest. Colab bootstrap (`_bootstrap`, L16) pip-installs packages at import and mounts Google Drive.
- Frontends call NCBI E-utilities (esummary/efetch/esearch) with `urllib`, no API key, no caching.
- No secrets committed (checked: `OPENAI_API_KEY` read from env/UI only).
