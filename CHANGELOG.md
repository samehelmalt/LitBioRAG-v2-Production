# Changelog

## 0.9.1 — 2026-10-31
- Light theme (persisted, follows the system preference by default).
- Token optimization: question-relevant evidence windows with offsets preserved, compact schema,
  generation cache keyed by exact evidence spans; per-answer token-saving report.
- Gate: off-topic veto (R6b) when evidence covers the asked entities but no critical claim names
  them; CLARIFY now judged on all retrieved evidence, not only cited passages.
- Calibration: `--target-precision` operating point (release automatically only above the
  confidence at which the calibration split reaches the target); `gate.yaml: operating_point`.
- docs/QUALITY_PLAYBOOK.md: the three numbers behind "90–98%", the levers in payoff order.

## 0.9.0 — 2026-10-31
First packaged pre-release of AEOS RAG (LitBioRAG v2 Production). Code complete for the
architecture; **no measured accuracy yet** (see docs/STATUS.md).
- Retrieval v2: Tantivy BM25 (stemmed + normalized + exact-symbol channel), FAISS dense, RRF,
  single-stage cross-encoder at token-level truncation, sentence-aligned passages with offsets.
- Model registry: every role is configuration (`configs/models.yaml`, env overrides); hosted
  generator profile behind an API key; family-independence rule; calibration bound to model id.
- Generation contract: atomic claims with citations, abstention reasons, conflicts; versioned
  abstain-first prompt; conversation context for follow-ups.
- Verification: NLI support/contradiction, citation resolution, entity consistency, question-entity
  grounding with confusable-entity detection (genes, drugs, conditions, numbered families;
  `configs/entities.yaml`), numeric checks, conflict records.
- Gate v2: twelve ordered rules → ANSWER / ANSWER_WITH_CAUTION / CLARIFY / ABSTAIN / ESCALATE,
  claim-level repair, evidence sufficiency, calibration job (uncalibrated until run).
- Human review queue: blocked/escalated answers await a reviewer; edited releases re-verified;
  decisions audited and exported as labels.
- Knowledge bases: upload PDF/DOCX/XLSX/PPTX/TXT/MD with page/section/sheet/slide citations;
  retrieval scope uploaded / indexed / both; entitlement before retrieval.
- Corpus index management: fetch, verify, activate, incremental update of PubMed and
  bioRxiv/medRxiv bundles; indexed-corpus switch.
- API: FastAPI with JWT and API keys, rate limits, audit log, request ids, SSE progress stream,
  chats, research tools, datasets, admin profile switch.
- UI: AEOS RAG dashboard (chat, documents, citations, analysis, source panel, all sidebar pages,
  trust page) rendering only API responses.
- Resources: token budgets, spend ledger with caps, GPU residency pool with pre-flight, caches.
- Evaluation harness with thesis baseline reproduction, split manifest, adversarial substitution
  suite (100% catch / 0% false alarm on the fixture, enforced in CI).
