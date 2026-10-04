# Retrieval Failure Analysis

**Status: NOT YET MEASURED on the production system.** This document is seeded with the retrieval failures observable in the thesis master-run CSVs (`Results/V5521_Run/`, read-only) and defines the classification every production failure will receive once the harness (Step 2) and the production baseline (Step 4) exist.

## Failure taxonomy (every failed query gets exactly one primary cause)

| Code | Cause | How it is established |
|---|---|---|
| QF | query formulation (long title-style question, missing key term) | gold doc retrieved by a hand-rewritten query |
| TERM | biomedical terminology / synonym mismatch | gold doc retrieved after MeSH/UMLS synonym expansion |
| ACR | acronym / abbreviation | gold doc retrieved after acronym expansion |
| PRE | preprocessing / tokenization (punctuation, hyphens, gene symbols) | gold doc retrieved by BM25 after analyzer change |
| IDX | indexing (document missing or malformed in index) | gold PMID absent from index or empty text |
| CHUNK | chunking / passage boundaries | gold passage split across units |
| BM25 | lexical retriever rank | gold doc in dense top-k, absent from BM25 top-k |
| DENSE | dense retriever rank | gold doc in BM25 top-k, absent from dense top-k |
| RRF | fusion rank | gold doc in one list's top-k but below fused cut |
| RERANK | reranker demotes gold doc | gold doc in pool, pushed out of top-k by the cross-encoder |
| EXP | query expansion hurt | gold doc retrieved without expansion, lost with it |
| COV | corpus coverage | gold doc not in corpus |

## Seed: zero-recall queries at S3a in the thesis run (medcpt × qwen_2.5_3b, k = 5)

| Query | S0 dense | S1 BM25 | S2 RRF | S3 two-stage | S3a one-stage | Provisional cause |
|---|---|---|---|---|---|---|
| Why is Trifluridine/Tipiracil more active than other fluoropyrimidines in colorectal cancer? | 0.667 | 0 | 0 | 0 | 0 | RRF/RERANK (dense had it; fusion with an empty BM25 side lost it) + PRE (slash-joined drug name for BM25) |
| Which properties of the mRNA does N6-methyladenosine (m6A) affect? | 0.25 | 0 | 0 | 0 | 0 | ACR/PRE (m6A, parentheses) + RRF |
| Should all cases of high-grade serous ovarian, tubal and primary peritoneal carcinomas be reclassified as tubo-ovarian serous carcinoma? | 1.0 | 1.0 | 1.0 | 0 | 0 | RERANK (gold doc in pool at S2, demoted by both reranking modes; 512-char truncation suspected) |
| Is Clostridioides difficile an aerobe or an anaerobe bacterium? | 0 | 0.5 | 0.5 | 0.5 | 0 | RERANK (one-stage CE demotes the BM25 hit) / TERM (Clostridioides vs Clostridium) |
| What is the role of metalloproteinase-17 (ADAM17) in NK cells? | 0 | 0 | 0 | 0 | 0 | ACR/TERM (ADAM17 vs "metalloproteinase-17"; gene-symbol tokenization) |
| Please list 2 treatments for a torn rotator cuff | 0 | 0 | 0 | 0 | 0 | QF (list-type question, no lexical anchor) |
| According to guidelines, insulin resistance is one risk factor in the diagnosis of metabolic syndrome, name 3 more risk factors. | 0 | 0 | 0 | 0 | 0 | QF (long compound question) |

Three of the seven were found by dense-only retrieval and then lost downstream; two were found by BM25 and lost by reranking. These are the first cases the production baseline must re-run with the taxonomy above.

## Planned measurements (Step 4 onward)

- Per query: rank of each gold document in BM25, dense, RRF and reranked lists; first relevant rank; which list first contained it.
- Per query type and per dataset: R@1/5/10/20, MRR, nDCG.
- Ablations: 512 chars vs full tokens in the reranker; one-stage vs different-model cascade; BM25 analyzer; RRF k and weights; conditional expansion with `retrieval_effect ∈ {positive, neutral, negative}` logged per query.
- Output: one row per failed query with primary cause code, evidence, and the fix that recovered it (if any).
