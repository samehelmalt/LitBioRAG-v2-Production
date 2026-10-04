# Thesis query sets (read-only copy)

Copied verbatim from `samehelmalt/LitBioRAG` at commit `ad0bacd0f3b878cd90f9a473bc79018223b27f1b` on 2026-10-04.

| Here | Origin | Records | Schema |
|---|---|---|---|
| `benchmark_24_queries.json` | `Results/calib/calib_1st_run/benchmark_24_queries.json` | 24 | `{query, pmids[]}` — the thesis evaluation set |
| `bioasq_deduped.json` | `benchmark_data/extracted queries/bioasq_deduped.json` | 1,562 | `{question, answer, Documents[pubmed URLs], decision, year, sourceFormat, questionId, queryType}` |
| `pubmed_deduped.json` | `benchmark_data/extracted queries/pubmed_deduped.json` | 2,000 | same as above (`decision` empty for PubMedQA summary items) |
| `balanced_queries_200.json` | `benchmark_data/extracted queries/balanced_queries_200.json` | 200 | same plus `dataset`, `dr_type`, `question_type` |

Gold PMIDs are given as `http://www.ncbi.nlm.nih.gov/pubmed/<pmid>` URLs; `evaluation/datasets/loaders.py` parses them. The 24 thesis queries are pinned to the `test` split by `evaluation/splits.py` so that nothing is tuned on them. Upstream licences: BioASQ (CC BY 2.5), PubMedQA (MIT).
