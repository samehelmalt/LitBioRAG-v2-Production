# UI prototype review (owner mockups, 2026-10-31) → Step 10 requirements

Reviewed: four mockups supplied by the owner — (01) "AEOS RAG — Biomedical RAG Research" chat
workspace, (02) "Acevantic EdTech Platform" LMS dashboard, (03) "AEOS Software Factory" multi-agent
dashboard, (04) "AuditOS" audit dashboard. Only (01) and (02) concern this product; (03) and (04)
are unrelated products and are not reviewed.

## Verdict
Mockup 01 is a good visual basis for the research UI and mockup 02 for the course product's
member/admin shell. Neither yet shows the trust surface that `docs/TRUST_AND_SAFETY.md` requires,
and one element (the Precision/Recall/Faithfulness/Relevance bar chart) must not ship as drawn.

## What mockup 01 already has that we keep
- Answer with numbered inline citations `[1][2][3]` → maps to `Claim.citations` rendered per claim.
- "Source Documents" panel with title, venue and a score → maps to `EvidenceUnit` (title, journal,
  year, `display_source`), but the score shown must be the reranker score labelled "retrieval
  relevance", never a correctness number.
- Tabs Chat / Documents / Citations / Analysis → Citations becomes the claim-by-claim verdict view;
  Analysis becomes the verification report, not a metrics chart.
- Sidebar items Knowledge Base, Upload Documents, Datasets, Model Settings, API Access, Evaluations
  → all exist as backend concepts (bundles, PPTX/corpus ingestion, split manifest, registry
  profiles, API keys, benchmark runs).

## Required changes (each is a trust requirement, not a style preference)
1. **Delivery action banner** at the top of every answer: ANSWER / ANSWER WITH CAUTION / CLARIFY /
   ABSTAIN / ESCALATE, with the deciding rule in plain words (from `Decision.trace`). An abstention
   is a first-class screen, not an empty chat bubble.
2. **Per-claim verdict** in the Citations tab: supported / unsupported / contradicted, the quoted
   evidence span (char offsets from `EvidenceUnit`), and the resolved citation. Unsupported claims
   removed by repair are listed under "Removed from this answer" with the reason.
3. **Entity grounding line**: the entities the question asked about and whether each was found in
   the supporting evidence; a near-neighbour substitution (MAP3K3 → MAP3K7) is shown as the reason
   for the abstention with both symbols highlighted.
4. **Mixed-evidence view**: when a `ConflictRecord` exists, both sides are shown side by side with
   population / design / year; the answer text says the evidence is mixed. No majority vote.
5. **Source tier flags** on every source card: peer-reviewed / preprint (not peer-reviewed) /
   course material / retracted-if-known, plus year. Preprints get a visible badge.
6. **Remove the metrics bar chart** (Precision / Recall / Faithfulness / Relevance per answer).
   None of these can be computed for a single live answer without gold labels; showing them
   implies a correctness the system cannot measure. Replace with the verification report fields
   (support rate, contradiction rate, citation resolution, entity grounding, sufficiency,
   calibrated confidence with "uncalibrated" shown until `calibrate_gate` has run).
7. **Footer on every answer**: model profile name and revisions (from `ModelRegistry.describe()`),
   corpus bundle id and date, prompt id, gate config version, "AI-generated; not medical advice",
   link to the public trust page.
8. **Model Settings** page is read-mostly: it shows the active profile from `configs/models.yaml`
   and lets an admin switch profile or set a role override; it never edits code. Calibration status
   per verifier is displayed (thresholds invalidated when the model changes).
9. **Evaluations** page shows benchmark runs from `evaluation/benchmarks/runs/` (retrieval,
   answer, adversarial substitution suite) with split names, so marketing numbers always trace to a
   run id.
10. **Feedback control** on each claim ("this is wrong / this is right") writing to the labelling
    JSONL protocol in `evaluation/labels.py`, so human labels accumulate from day one.

## Course product (mockup 02)
Keep the LMS shell (courses, lectures, progress) as the member area; the chat view is the same
component as mockup 01 with slide citations rendered as "Lecture title, Slide N" and a slide
preview behind entitlement. Add: course selector (drives `course_ids` filter), content version
shown on each citation, ESCALATE screen text for patient-specific questions, admin upload with the
deck diff report before re-index.

## Build order
Step 9 API first (the UI is a thin client over `/api/v1/query`); Step 10 implements the two screens
above from the API response only — no UI logic recomputes a verdict.
