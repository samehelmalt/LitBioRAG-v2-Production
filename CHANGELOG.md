# Changelog

## 0.10.0 — 2026-11-02
Closes the gaps of the external v0.9.2 assessment and the LitBioRAG failure classes.
- Retrieval: paired MedCPT query/article encoders (searches use the query tower; registry
  refuses a mismatched pair).
- Generation: model-family profiles (Qwen3 thinking off, DeepSeek reasoning-tag stripping,
  decoding controls, stop strings); output-integrity checks (repetition, n-gram loops, role
  echoes, truncated JSON, duplicate claims, uncovered sentences, off-topic); bounded regeneration
  with precise evidence-bound feedback and an immutable attempt ledger.
- One transaction, six channels: RAW / CANDIDATE / USER / AUDIT / CONTROL / HISTORY records per
  request id (`GET /api/v1/records/{id}`); chats keep delivered turns only; metrics reported as
  candidate / verified / released.
- Conversation: follow-ups rewritten from the delivered state and shown as "Interpreted as";
  earlier answers are context, never evidence.
- Slides: OCR (Tesseract) and hosted vision captions, both optional; image-only slides flagged
  `needs_ocr`; lecture provenance (course / lecture / year / version) on upload; canonical slide
  ids; `GET /api/v1/slides/{file_id}/{n}`; deck versions supersede in search while old citations
  still resolve.
- Knowledge bases: incremental FAISS id-mapped add/remove and Tantivy deletes with atomic
  persistence; persistent index jobs.
- Education policy: lecture-first with labelled `lecture_vs_literature` conflicts; education
  profile defaults to the uploaded scope; course-material share in evidence selection.
- Evaluation: lecture QA question-set schema, synthetic 30-slide prototype deck and 34-question
  set, slide-level runner (Recall@k, MRR, citation validity, delivered precision, false PASS,
  unanswerable abstention, defects, latency, cost); span-in-prompt check per claim.
- Platform: embeddable chat widget (`/embed`, `static/widget.js`), tenant-scoped identities,
  `local-8b` and `local-14b` model profiles.

## 0.9.2 — 2026-10-31
- Product name AEOS Medical; medical-cross logo.
- Docker local test run: `docker compose up --build` builds a demo bundle on first start, issues
  demo tokens, serves on port 8471; `scripts/entrypoint.py`; data volume.

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
First packaged pre-release of AEOS Medical (LitBioRAG v2 Production). Code complete for the
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
- UI: AEOS Medical dashboard (chat, documents, citations, analysis, source panel, all sidebar pages,
  trust page) rendering only API responses.
- Resources: token budgets, spend ledger with caps, GPU residency pool with pre-flight, caches.
- Evaluation harness with thesis baseline reproduction, split manifest, adversarial substitution
  suite (100% catch / 0% false alarm on the fixture, enforced in CI).
