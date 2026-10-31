# API (v1)

Run: `python -m scripts.serve --bundle data/artifacts/<id> --profile hosted --port 8471`
(settings in `configs/api.yaml`; secrets only from environment variables; OpenAPI at `/api/docs`).
The service and the demo script share one code path, `app/service/pipeline.py`, so what the API
returns is exactly what the evaluation runners measure.

## Endpoints
| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | status, bundle id, profile, gate calibration status, GPU pre-flight, today's API spend |
| GET | `/api/v1/models` | principal | active profile with model names, families, quantization, calibration status, gate costs |
| POST | `/api/v1/query` | principal | question → delivery decision with the full trust surface |
| GET | `/api/v1/sources/{doc_id}` | principal | source document metadata and abstract; slides require course entitlement |

A *principal* is a JWT bearer token, an API key (`X-API-Key`) or, when `auth.required` is false,
an anonymous caller limited to the public corpus. See SECURITY.md.

## POST /api/v1/query
Request: `{"question": str, "course_ids": [str] | null, "max_units": int | null}`.

Response fields and what the UI must do with them (UI_PROTOTYPE_REVIEW.md):
- `action` ∈ ANSWER / ANSWER_WITH_CAUTION / CLARIFY / ABSTAIN / ESCALATE, with `rule` (the
  deciding gate rule, e.g. `R6-entity`), `reason` (plain words) and `trace` (every rule's line).
  Render as the banner; an ABSTAIN or ESCALATE has its own screen.
- `confidence`, `confidence_calibrated` (false until the calibration job has run: show the word
  "uncalibrated"), `sufficiency`.
- `answer`: the delivered text after claim-level repair. `removed_claims`: what was removed.
- `claims[]`: per-claim verdict (`label`, `support`, `contradiction`), citations, `supporting_ids`,
  `fabricated_ids`, `entity_missing`, `numeric_mismatches`, `substitution` (claim entity, asked
  entity, rule such as `numbered_family` or `drug_class:statin`), `off_target`, `removed`,
  `evidence_span` / `contradicting_span` (passage id and char offsets into the source text).
- `conflicts[]`: claims with both support and contradiction; render both sides, never a vote.
- `entities`: what the question asked about, what was missing in the evidence or in the claims,
  and `substitution_detected`.
- `sources[]`: provenance for every evidence unit (citation id, title, year, journal, `tier` =
  peer_reviewed / preprint / course_material / other, url, text with offsets, retrieval score,
  rank, and lecture / slide fields for course material). The retrieval score is a relevance
  score, never a correctness score.
- `usage` (prompt / completion tokens, exact or estimated, cost, latency), `evidence_fit` (kept /
  truncated / dropped passages under the token budget).
- `versions`: profile, model names and revisions, bundle id and date, prompt id, gate
  calibration date, API version. Render in the footer of every answer.
- `disclosure`: AI disclosure text; render verbatim.

Errors are `{"request_id", "error"}` with 401 / 403 / 404 / 422 / 429 (`Retry-After`) / 500
("internal error", never a stack trace). Every response carries `X-Request-ID`.

## Observability
One structured log line per request (`event: access`) and one audit JSON line per query
(`configs/api.yaml: audit.path`) with request id, principal, courses, question (or its hash in
privacy mode), action, rule, counts, usage, stage timings and versions. The usage ledger
(`COST_AND_RESOURCES.md`) records every hosted model call.


## Human review (reviewer role)
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/review/queue?status=pending|released|blocked|expired|all` | cases (summary) |
| GET | `/api/v1/review/stats` | counts by status |
| GET | `/api/v1/review/{case_id}` | full trust surface of one case |
| POST | `/api/v1/review/{case_id}/decision` | `{decision: release|block, edited_answer?, kept_claim_ids?, notes, confirm}`; 409 with `reverification` when an edited release has flagged claims and `confirm` is false |
| GET | `/api/v1/my/cases` | the asker's own cases and released answers (signed-in users) |
A query whose answer was withheld carries `review: {case_id, status: "under_review", message}`.

## Knowledge bases (uploaded files) and retrieval scope
`POST /api/v1/query` accepts `scope` (`indexed` | `uploaded` | `both`) and optional `kb_ids`; the
response echoes `scope` and the `kb_ids` searched. Uploaded sources carry `kb_id`, `file_id`,
`file_name`, `locator` (`p3`, `s7`, `Sheet1!A2:B41`, `sec2`) and `locator_label`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/kb` | knowledge bases the caller can read, with files |
| POST | `/api/v1/kb` | create (`{name, description}`) |
| GET | `/api/v1/kb/{id}` / `/status` | details; file statuses and running index jobs |
| POST | `/api/v1/kb/{id}/upload` | multipart `file` (PDF/DOCX/XLSX/PPTX/TXT/MD; size limit from api.yaml), `background=true` for async indexing |
| POST | `/api/v1/kb/{id}/share` | `{subs: [...]}` read access for other users |
| DELETE | `/api/v1/kb/{id}/files/{file_id}` · `/api/v1/kb/{id}` | remove a file / the whole base (units leave the index immediately) |

## Corpus indexes (admin role)
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/corpus` | bundles on disk (id, created, counts, encoder, updates, active flag), enabled switch, jobs |
| POST | `/api/v1/corpus/activate` | `{corpus_id}` verify + swap the served bundle |
| POST | `/api/v1/corpus/enabled` | `{enabled}` serve the indexed corpus or uploads only |
| POST | `/api/v1/corpus/jobs` | `{type: fetch|verify|update|fetch_preprints|rebuild, params}` background job |
| GET | `/api/v1/corpus/jobs/{id}` | status, result, log tail |
