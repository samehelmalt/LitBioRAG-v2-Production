# Trust and Safety

**Status: the verification layer exists and is tested with deterministic doubles; the gate (Step 8) and its calibration do not exist yet. No false-PASS rate has been measured for the production system.** This document says what each check catches, what it cannot, and how the delivery decision will be made and reported.

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
| Entity consistency | entity substitution in the answer (MAP3K3 → MAP3K7), entities absent from the supporting evidence | relation errors between correct entities (A inhibits B vs B inhibits A); synonyms the regex extractor does not know (scispaCy/UMLS adapter planned) |
| Numeric consistency | altered values, wrong unit class (61 % vs 61 patients), numbers absent from supporting evidence | a correct number attached to the wrong group or outcome (tuple-level role checking is the next refinement) |
| Conflict detection | one source supports and another contradicts the same claim | conflicts that only emerge across claims, or from evidence that was not retrieved |
| Contract parser | unparseable or schema-violating output → forced abstention | — |
| Prompt boundary | passage text rendered as data; instructions inside passages do not reach the system prompt | a model that follows injected instructions anyway → verification still scores the claims against evidence, so an injected "supported" cannot be asserted by the model |

## Delivery decision (Step 8, not yet implemented)

Ordered vetoes, then one calibrated decision. No composite.

```text
scope / privacy veto            → ESCALATE or ABSTAIN
citation sufficiency            → every critical claim cited and resolved, else ABSTAIN
contradiction veto              → any critical claim contradicted, else ABSTAIN (conflict → "evidence is mixed")
entity / relation veto          → critical claim with missing entity, else ABSTAIN
numeric veto                    → critical claim with numeric mismatch, else ABSTAIN
unsupported-claim threshold     → above threshold ABSTAIN; non-critical unsupported claims removed, then ANSWER_WITH_CAUTION
evidence sufficiency            → below threshold ABSTAIN or CLARIFY
calibrated confidence           → ANSWER vs ANSWER_WITH_CAUTION operating point
```

Actions shown to users: **ANSWER**, **ANSWER WITH CAUTION**, **CLARIFY**, **ABSTAIN**, **ESCALATE**. Internally the gate state is still recorded as PASS/WARN/BLOCK for comparability with the thesis rows. Every decision carries the policy trace that produced it.

Thresholds are fitted on the `calibration` split against labels (proxy labels until expert labels exist) with an explicit expected-cost objective (`configs/gate.yaml`: c_false_pass, c_false_block, c_warn, c_abstain). They are written by the calibration job and attached to the model they were fitted for in `configs/models.yaml`; a swapped model has no thresholds until recalibrated.

## What is reported (never as one number)

coverage · selective risk · false PASS · false WARN · false BLOCK · abstention precision/recall · ECE · Brier · coverage–risk curve and risk at target coverage · per-check rates (citation resolution, citation support, entity fidelity, numeric fidelity, contradiction rate, conflict rate). Definitions: `evaluation/metrics/trust.py`, `docs/EVALUATION.md`.

## Labels

Proxy labels (yes/no polarity correct ∧ gold document retrieved ∧ unsupported ≤ 0.25 ∧ no contradiction) until a human-labelled set exists; proxy-based numbers are lower bounds on acceptability and are always marked "proxy". The export/import format for human labelling is in `evaluation/labels.py`.

## Scope

LitBioRAG supports biomedical learning and literature research. It does not give patient-specific diagnosis, treatment or dosing advice; such questions are routed to ESCALATE with a fixed message. Users are told they are interacting with an AI system and see model and corpus versions with every answer.
