# Changelog

## 0.12.0 — 2026-11-02
Public landing page and self-service demo.
- `GET /` is now a landing page (hero with institution and student paths, six-step how-it-works
  diagram, feature grid, real screenshots, claim / do-not-claim block, deployment shapes, trial
  request form); the application moved to `/app`. Copy is owner-editable in `configs/landing.yaml`
  (`GET /api/v1/landing`); no prices or accuracy figures are invented.
- Leads: `POST /api/v1/leads` (validated, honeypot, rate limited, JSONL + optional webhook),
  `GET /api/v1/leads` for admins. No e-mail sending.
- Demo access: `POST /api/v1/demo/token` mints 2-hour visitor tokens when `AEOS_DEMO_TOKENS=1`
  (default in `docker-compose.yml`, refused in client mode); `/app?demo=1` signs in automatically
  and shows a demo banner; the trial allowance applies.
- `scripts/screenshots.py` + `scripts/screenshots.js` (Node Playwright) regenerate the landing
  screenshots from a running server; `scripts/release.py --screenshots-from <url>`.

## 0.11.0 — 2026-11-02
Concurrency for many simultaneous users (owner question: "Do we need vLLM for thousands of
students?").
- Admission control per model role (`app/core/admission.py`): capacity, FIFO queue with the
  position shown on the progress stream, bounded queue and timeout → `503` + `Retry-After`
  (trial refunded, audit line); counters in `/health.concurrency`; `AEOS_CONCURRENCY_<ROLE>`.
- Cross-request micro-batching for the verifier and the cross-encoder reranker
  (`app/core/batcher.py`): concurrent requests share one forward pass; a lone request never waits;
  scores identical to direct calls. `CrossEncoderReranker.score_pairs` for mixed-query batches.
- Redis-backed rate limiter, trial ledger, usage ledger (shared daily cap) and KB job table
  (`app/core/shared_state.py`; `configs/api.yaml: state_backend: redis`, `REDIS_URL`); `scale`
  extra; `fakeredis[lua]` in dev.
- `vllm-8b` profile (generator behind vLLM's OpenAI-compatible endpoint, `VLLM_URL`),
  `compose.gpu-vllm.yaml` (vLLM + API + Redis on one GPU host), `compose.scale.yaml` (N API
  replicas behind nginx + Redis), `deploy/nginx.conf`.
- `scripts/load_test.py`: 10/50/100 concurrent users, p50/p95/p99, status mix, cache hits,
  generator tokens/s → `evaluation/benchmarks/runs/<name>/load/`; DEPLOYMENT.md "Scaling" with
  capacity estimates marked NOT MEASURED.

## 0.10.1 — 2026-11-02
Ported from the owner's "Cloud Selector v1.4" fork, adapted to the v0.10.0 base:
- Trial allowance per authenticated identity (reserve-before-run, refund-on-failure, idempotent
  per request id, persistent, admin-exempt) and `GET /api/v1/trial`.
- Client-mode start-up guard (`AEOS_CLIENT_MODE=1`): refuses the test profile, the demo bundle and
  a short or per-run JWT secret; prospect token issuer `scripts/issue_client_token.py`.
- Native PowerPoint chart data (series and category values) in slide text, marked as read from
  the chart object, never a pixel estimate.
- `compose.client-trial.yaml` (localhost-bound, authenticated, hashed audit) and
  `compose.offline-demo.yaml` with the `offline-demo` profile of real small CPU models and a
  weight prefetch/verify script; `WITH_OCR` Docker build argument; `ocr` extra.
- `cloud-anthropic` and `cloud-deepseek` profiles that keep MedCPT retrieval and the large verifier.
Not ported on purpose: the credential-setting HTTP endpoint, the environment-variable OCR switch
that wrote placeholder strings into indexed text, and cloud profiles built on general-domain MiniLM.

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
