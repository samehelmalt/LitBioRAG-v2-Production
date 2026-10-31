# Governance: how a change reaches users

## Registries (what is versioned)
| Registry | Where | Changes by |
|---|---|---|
| Models | `configs/models.yaml` (kind, id, revision, quantization, calibration) | profile edit + benchmark entry in MODEL_SELECTION.md |
| Prompts | `app/generation/prompts.py` (`PROMPT_ID`) | new id; GENERATOR_FAILURE_ANALYSIS.md entry |
| Thresholds | `configs/gate.yaml`, `models.yaml: calibration` | `scripts/calibrate_gate.py` only; hand edits are rejected in review |
| Entity rules | `configs/entities.yaml` | PR with suite result (`scripts/run_adversarial.py`) |
| Corpus bundles | `artifacts/<corpus_id>/manifest.json` | build host job; served bundle id appears in every answer |
| Datasets / splits | `evaluation/datasets/splits/manifest_v1.json` | new manifest version; never edit v1 |
| Resources | `configs/resources.yaml`, `configs/pricing.yaml` | PR; pricing dated |

## Change control checklist (every PR)
1. CI green (`ruff`, `pytest`, including the substitution suite at 100% catch / 0% false alarm).
2. If a model changed: calibration invalidated and re-run; MODEL_SELECTION.md row added with the
   split, metrics and cost column; family-independence rule still holds.
3. If a prompt changed: new `PROMPT_ID`; answer baseline re-run; false-PASS not worse.
4. If thresholds changed: produced by the calibration job on the `calibration` split; coverage–risk
   curve attached; false PASS reported at the chosen operating point.
5. If the corpus changed: bundle manifest verified; retrieval baseline re-run; bundle id bumped.
6. Docs updated in the same PR (the PRODUCTION_AUDIT.md section that the change affects).

## Release gates (public app)
- Measured BASELINE_RESULTS.md with false PASS on the `test` split and the adversarial suites.
- Gate calibrated (`gate.yaml: calibrated_on` non-empty) for the served verifier.
- Trust page published (versions, metrics, known failure modes, labelling status).
- Security review items in SECURITY.md closed or explicitly accepted by the owner.

## Incident handling
Any confirmed wrong delivered answer is filed with request id (audit log), versions and the
claim-level verdicts; a regression case is added to the adversarial suites before the fix ships.


## Reviewer duties
Reviewers decide only on cases the gate blocked; they never see or alter delivered answers. A
release must be justified in `notes`; an edited release that fails re-verification may be confirmed
only when the text is generic safety guidance that the evidence cannot and need not support
(for example "consult your clinician"). Reviewer decisions are sampled weekly against a second
reviewer; disagreement rate is reported on the trust page once labels exist.
