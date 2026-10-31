# Trust and Safety

**Status: the verification layer and the gate exist and are tested with deterministic doubles. The gate runs on heuristic defaults marked UNCALIBRATED in `configs/gate.yaml`; the calibration job exists but has not been run, because no real bundle or model has been run. No false-PASS rate has been measured for the production system.** This document says what each check catches, what it cannot, and how the delivery decision is made and reported.

## Principle

The evidence is authoritative; the generator is not. An answer is released only when every important claim is supported by evidence the system actually retrieved, no important claim is contradicted, entities and numbers match the supporting passages, and every citation resolves to a passage that supports its claim. Abstaining is a correct outcome. The primary safety metric is **false PASS**: released answers that contain unsupported, contradicted or mis-cited claims.

## What the thesis did that is deliberately not carried forward

| Thesis behaviour | Why it is gone |
|---|---|
| One linear composite of six signals, PASS at ≥ 0.28 | three of the six were the same NLI classification re-weighted; a 0.20 weight for "no numbers present" and a near-noise retrieval signal could carry an answer with 44 % unsupported claims to PASS (PRODUCTION_AUDIT.md §9) |
| `Post_Gate_Faith = 0.90` for blocked answers | an abstention is an action, not a faithful answer; the constant produced the headline +0.452 |
| gold yes/no label inside the gate's NLI signal | not available at runtime; `app/` imports nothing from `evaluation/` and the CI test suite checks it |
| "MUST NOT abstain", majority vote on conflicting passages | forced answering and vote-away of disagreement are the opposite of the contract |
| substring match of digits, regex query terms vs context | replaced by unit-class numeric matching and entity consistency against *supporting* passages |

## Checks in the verification layer (`app/verification/`)

| Check | Catches | Cannot catch (known limits) |
|---|---|---|
| Citation resolution | ids not in the evidence set (fabricated PMIDs/DOIs); cited passages that do not entail the claim | a real passage cited for the wrong sentence of a multi-sentence claim when both sentences are in it |
| Entailment / contradiction (verifier) | unsupported claims; claims contradicted by a passage; a confident NEUTRAL no longer hides an ENTAILMENT in another chunk | paraphrases the verifier model misjudges; depends on verifier quality → benchmark in Step 7, calibration in Step 8 |
| Entity consistency | entities in a claim absent from its supporting evidence | relation errors between correct entities (A inhibits B vs B inhibits A); synonyms the regex extractor does not know (scispaCy/UMLS adapter planned) |
| Question-entity grounding + substitution | a correctly cited claim about the **wrong** entity, for genes, drugs, conditions and numbered families alike (MAP3K7 for MAP3K3, simvastatin for atorvastatin, hyperkalemia for hypokalemia, type 1 for type 2 diabetes, hepatitis C for hepatitis B, hydralazine for hydroxyzine): the question's entities must appear in the supporting evidence and in the critical claims; a near neighbour in their place is a substitution and a hard veto. Six rules decide "near neighbour", all configured in `configs/entities.yaml`: numbered family (digits differ), letter variant, same-stem edit, antonym prefix (hypo/hyper, brady/tachy), shared drug-class suffix (INN stem), explicit confusable group (ISMP-style look-alike list). Aliases (MEKK3=MAP3K3, TAK1=MAP3K7, heart attack=MI, T2DM=type 2 diabetes) are normalized, not flagged; generic vs specific (hepatitis / hepatitis B, COVID / COVID-19) is never flagged. Retrieval adds an exact-symbol channel so a near-neighbour document gains nothing from the asked symbol | substitutions between entities with unrelated names that are not in the confusable list (ontology typing with scispaCy/UMLS extends the list automatically: Step 5e); relation direction |
| Numeric consistency | altered values, wrong unit class (61 % vs 61 patients), numbers absent from supporting evidence | a correct number attached to the wrong group or outcome (tuple-level role checking is the next refinement) |
| Conflict detection | one source supports and another contradicts the same claim | conflicts that only emerge across claims, or from evidence that was not retrieved |
| Contract parser | unparseable or schema-violating output → forced abstention | — |
| Prompt boundary | passage text rendered as data; instructions inside passages do not reach the system prompt | a model that follows injected instructions anyway → verification still scores the claims against evidence, so an injected "supported" cannot be asserted by the model |

### Substitution suite (adversarial regression, runs in CI)

`evaluation/adversarial/substitution.py` generates attack/control pairs from any corpus: for every two
documents whose titles carry near-neighbour symbols (MAP3K3/MAP3K7, MAP3K3/MAPK3, ...), the attack
answers a question about A with a true, correctly cited claim from B; the control answers it from A.
Release rule: catch rate 1.0 on attacks and false-alarm rate 0.0 on controls for the fixture suite
(`tests/security/test_substitution_suite.py`); for every real bundle and verifier the same two numbers
are reported in MODEL_SELECTION.md via `python -m scripts.run_adversarial --bundle … --profile …`,
which exits non-zero when the catch rate is below 1.0. A substitution is caught by the entity layer
alone, so it does not depend on which NLI verifier is configured. The summary breaks catch rate down by
rule (`by_reason`) and by entity kind (`by_kind`: gene / drug / condition), so a regression in one
class is visible even when the overall rate looks fine. To extend coverage, add a group, synonym,
prefix pair or suffix to `configs/entities.yaml`; no code changes.

## Delivery decision (Step 8, not yet implemented)

Ordered vetoes, then one calibrated decision. No composite.

Implemented in `app/safety/gate.py` (`decide`). Rules fire in this order; the first that fires decides and is named in the trace:

```text
R1  scope: patient-specific / personal dosing / emergency            → ESCALATE (fixed message)
R2  generator abstained                                               → ABSTAIN (its reason)
R3  any fabricated citation                                           → ABSTAIN
R4  critical claim uncited, or its citation does not support it       → ABSTAIN
R5  critical claim contradicted                                       → ABSTAIN ("conflicting" wording if sources disagree)
R6  entity substitution (near-neighbour of a question entity) or
    critical-claim entity mismatch                                    → ABSTAIN  (never WARN)
R6b evidence covers the asked entities but no critical claim names any → ABSTAIN (off-topic)
R7  critical numeric mismatch                                         → ABSTAIN
R8  question entities absent from the evidence                        → CLARIFY
R9  evidence sufficiency below threshold                              → ABSTAIN
R10 non-critical unsupported claims removed (claim-level repair,
    removed claims listed); too many → ABSTAIN                        → ANSWER_WITH_CAUTION
R11 a conflict record on any claim                                    → ANSWER_WITH_CAUTION
R12 calibrated confidence ≥ τ                                         → ANSWER, else ANSWER_WITH_CAUTION
```

Actions shown to users: **ANSWER**, **ANSWER WITH CAUTION**, **CLARIFY**, **ABSTAIN**, **ESCALATE**. Internally the gate state is still recorded as PASS/WARN/BLOCK for comparability with the thesis rows. Every decision carries the policy trace that produced it. Evidence sufficiency (`app/safety/sufficiency.py`) uses runtime features only: supporting documents, question-entity grounding, BM25/dense agreement, exact-symbol hits. Score spread is not a feature. Until `scripts/calibrate_gate.py` has run, confidence is the minimum of the check scores (the weakest check bounds it) and every report says UNCALIBRATED.

Thresholds are fitted on the `calibration` split against labels (proxy labels until expert labels exist) with an explicit expected-cost objective (`configs/gate.yaml`: c_false_pass, c_false_block, c_warn, c_abstain). They are written by the calibration job and attached to the model they were fitted for in `configs/models.yaml`; a swapped model has no thresholds until recalibrated.

## What is reported (never as one number)

coverage · selective risk · false PASS · false WARN · false BLOCK · abstention precision/recall · ECE · Brier · coverage–risk curve and risk at target coverage · per-check rates (citation resolution, citation support, entity fidelity, numeric fidelity, contradiction rate, conflict rate). Definitions: `evaluation/metrics/trust.py`, `docs/EVALUATION.md`.

## Labels

Proxy labels (yes/no polarity correct ∧ gold document retrieved ∧ unsupported ≤ 0.25 ∧ no contradiction) until a human-labelled set exists; proxy-based numbers are lower bounds on acceptability and are always marked "proxy". The export/import format for human labelling is in `evaluation/labels.py`.

## Scope

LitBioRAG supports biomedical learning and literature research. It does not give patient-specific diagnosis, treatment or dosing advice; such questions are routed to ESCALATE with a fixed message. Users are told they are interacting with an AI system and see model and corpus versions with every answer.


## Human review of blocked cases (`app/review/`)
The machine gate decides first and still blocks. A human can only add a release, never bypass the
gate silently:
- Every ABSTAIN with internal state BLOCK (except abstentions nobody can fix: no evidence, resource
  limit) and every ESCALATE creates a review case (`configs/review.yaml`). The asker sees
  "under review" and finds the outcome under My cases; nothing is delivered until a reviewer acts.
- Reviewers (JWT role `reviewer` or `admin`) see the full trust surface of the case and choose
  **release** (the machine's supported claims, a subset of them, or an edited text) or **block**.
- An edited release is **re-verified against the same evidence** by the served verifier. If any
  claim is unsupported, contradicted or substitutes an entity, the release is refused (409) with
  the re-verification shown; the reviewer may then confirm explicitly. The confirmation, the
  reviewer's identity and the flags are recorded.
- Every decision is audited and exported as a label record (`labels_from_review.jsonl`,
  LabelRecord shape) with the machine action and rule, so human decisions feed calibration and the
  false-PASS / false-BLOCK measurement. Pending cases expire after `expire_hours`.
