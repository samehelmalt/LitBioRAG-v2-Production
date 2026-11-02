# Status — what is done, what is missing (2026-11-02, v0.10.0)

## Done (code, tests, docs; all in CI)
| Area | State |
|---|---|
| Retrieval v2 (BM25 + dense + RRF + single-stage reranker, exact-symbol channel) | done; measured only on the fixture |
| Model registry, hosted profile, family-independence, calibration binding | done |
| Generation contract, prompts, hosted/local adapters, token budget, spend ledger | done |
| Verification: NLI, citations, entities (confusable detection for genes/drugs/conditions), numbers, conflicts | done |
| Gate v2 (12 rules, 5 actions, repair, sufficiency), calibration job | done; **uncalibrated** |
| Human review queue with re-verified releases, audit, label export | done |
| Knowledge bases: PDF/DOCX/XLSX/PPTX/TXT/MD upload, incremental index, scope selector | done |
| Corpus index management: fetch/verify/activate/update, indexed switch | done |
| API (auth, entitlements, limits, audit, SSE, chats, keys, tools, admin) | done |
| AEOS Medical UI (all sidebar pages, trust page) | done; rendered from API only |
| Evaluation harness, thesis baseline reproduction, split manifest, adversarial suite | done |
| Packaging: release script, Dockerfile, docker-compose, CHANGELOG | done |
| Paired MedCPT query/article encoders (thesis used the article tower for queries) | done |
| Generator profiles (Qwen3 / DeepSeek controls), output-quality checks, bounded attempts | done |
| Immutable answer records with RAW/CANDIDATE/USER/AUDIT/CONTROL/HISTORY channels | done |
| Conversation memory from delivered turns only; follow-up rewriting shown as "Interpreted as" | done |
| Slide images via OCR / hosted vision (optional); image-only slides flagged | done; no engine bundled |
| Lecture provenance, canonical slide ids, slide endpoint, deck versioning | done |
| Incremental KB index (add/remove, atomic persistence, persistent jobs) | done |
| Lecture-first policy with labelled literature differences | done |
| Lecture QA harness + synthetic 30-slide prototype deck | done; measured only with test doubles |
| Embeddable widget, tenant isolation, local-8b / local-14b profiles | done |

## Missing / blocked (honest list)
1. **Measured baseline (E4).** Needs an API key in the serving environment (hosted profile) or a
   GPU host, plus a real corpus bundle built on the build host. Until then: no accuracy, no
   false-PASS rate, no latency, no cost per answer. Trust page shows NOT YET MEASURED.
2. **Model benchmarks and calibration** (Steps 5, 6, 8-calibration) depend on 1.
3. **Full PubMed index** is a build-host job (days on one GPU); the serving host only fetches and
   updates. The daily-update fetcher for PubMed needs the NCBI file list wiring in the UI job
   (parse path exists; `update` job accepts the parsed JSONL).
4. **OCR / vision engines** are optional installs (Tesseract binary + pytesseract; a vision API
   key); the pipeline flags image-only slides until one is enabled.
5. **Ontology extractor** (scispaCy/UMLS) to extend confusable detection beyond the rules and the
   curated list.
6. **Multi-replica deployment**: rate limiter, ledger, job runner and caches are single-process;
   Redis/Postgres variants are not built.
7. **Security**: no penetration test yet; secret rotation runbook; dependency scanning in CI.
8. **Reviewer notifications** (email/webhook) — log-only hook.
9. **Human labels**: zero until users and reviewers produce them; proxy labels until then.
10. **Prototype numbers**: the lecture QA harness has only run with deterministic test doubles
    (pipeline checks); real-model results require item 1.
11. **Legal/branding**: "AEOS Medical" name and logo are placeholders created for this build; the owner
    must confirm trademark clearance before public launch.
