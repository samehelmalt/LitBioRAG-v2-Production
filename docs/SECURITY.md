# Security

## Threat model (what the service defends against)
1. **Prompt injection through retrieved text.** Passages are rendered as data inside a fixed,
   versioned prompt; the output must be schema-valid JSON whose citations resolve to the evidence
   set; every claim is verified against the cited passage; the gate decides delivery. A passage that
   says "ignore previous instructions" can at most produce an unsupported claim, which is removed
   or abstained (tests: `tests/security/test_adversarial_evidence.py`, `tests/api`).
2. **Fabricated or wrong-entity answers.** Citation resolution, entity grounding with confusable
   detection, numeric checks, hard vetoes in the gate (TRUST_AND_SAFETY.md).
3. **Unauthorized access to paid course material.** Entitlement is enforced before retrieval
   (`course_ids` filter in BM25 and dense search) and again on `/sources/{id}` for slides; a
   request for an unheld course is a 403, never a silent narrowing.
4. **Credential leakage.** API keys and JWT secrets are read from environment variables named in
   config; never from config files, request bodies or the UI. Logs never contain tokens. The
   repository boundary rules (REPOSITORY_BOUNDARY.md) forbid committing secrets; `.env` is ignored.
5. **Abuse and cost attacks.** Per-principal rate limiting (token bucket; Redis in multi-replica
   deployments), question length limits, evidence budget, per-request and per-day spend caps in
   the usage ledger; a crossed cap abstains with reason `resource_limit`.
6. **Information leakage in errors.** Unhandled errors return `internal error` with a request id;
   token failures return one generic message.

## Authentication and authorization
- JWT (HS256 default; the algorithm list is configuration) with claims `sub`, `courses`, `roles`,
  `aud`. Issued by the owner's identity provider or LMS; this service only verifies.
- API keys for server-to-server callers (`LITBIORAG_API_KEYS="key:subject,..."`), compared in
  constant time; no course entitlements.
- `roles: ["admin"]` may query any course (for faculty consoles).
- Anonymous access only when `auth.required: false`, limited to the public corpus.

## Not yet done (tracked)
- Redis-backed rate limiting and ledger for multi-replica deployments.
- Secret rotation runbook and dependency vulnerability scanning in CI.
- Penetration test of the deployed UI + API before public launch.
