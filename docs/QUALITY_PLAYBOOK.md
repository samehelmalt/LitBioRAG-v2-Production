# Quality playbook: how this product reaches, measures and keeps 90–98%

The owner's target is "more than 90 to 98%". Three different numbers hide behind that sentence,
and the product controls them differently. This document says which is which, what has been
built for each, and what remains honest to claim.

## 1. Three numbers, not one

| Number | Definition | Who controls it |
|---|---|---|
| **Delivered-answer precision** | share of answers delivered as ANSWER (and ANSWER_WITH_CAUTION) whose claims are all correct and correctly cited | the gate's operating point; this is the number the product promises users |
| **Coverage** | share of questions that receive an automatic answer rather than CLARIFY / ABSTAIN / review | the same operating point, in the opposite direction |
| **Model accuracy** | how often the generator's raw answer is right before any gate | model choice, retrieval quality, prompts |

A selective system can hold delivered precision at 95–98% *by construction*: it releases only
the answers whose calibrated confidence is above the threshold at which the calibration set shows
that precision, and sends the rest to caution or to a human reviewer. What the model accuracy
decides is the coverage you get at that precision, not the precision itself. So "98%" is a
setting the owner chooses on the coverage–risk curve; raising model accuracy raises coverage.

`scripts/calibrate_gate.py --target-precision 0.95` picks exactly that threshold (lowest
confidence at which released answers reach the target on the calibration split, maximum
coverage), records the achieved precision and coverage in `configs/gate.yaml: operating_point`,
and the trust page shows both. If the target is unreachable on the calibration rows, nothing is
released automatically and every answer goes to caution/review — the system never claims a
precision it has not seen.

## 2. The levers, in the order they pay off

1. **Measure first (E4).** Run the retrieval, answer and adversarial baselines with the real
   profile and bundle. Until then no lever can be evaluated. Blocked on the API key / GPU host.
2. **Retrieval recall.** Every answer is bounded by what is retrieved. Benchmark encoders
   (MedCPT vs BGE-M3 vs NV-Embed class), reranker (bge-reranker-v2-m3, MedCPT-CE), RRF weights,
   exact-symbol boost, and query understanding (MeSH/UMLS expansion with retrieval effect logged).
   Fix the failure classes in `RETRIEVAL_FAILURE_ANALYSIS.md` by count, largest first.
3. **Verifier quality.** The verifier decides what counts as supported. Benchmark DeBERTa-v3-large
   NLI vs MiniCheck-class vs a hosted judge of a different family; measure against human labels
   from the review queue and feedback. A better verifier moves false PASS down directly.
4. **Generator.** Benchmark 4-bit local 8–14B vs hosted candidates on the structured contract:
   format validity, citation correctness, claim support rate, entity substitution rate, cost per
   1,000 answers. Pick one winner per role (`MODEL_SELECTION.md`).
5. **Calibration.** Fit confidence and sufficiency on the calibration split; choose the operating
   point (target precision or expected cost); re-fit whenever a model changes (the registry
   invalidates thresholds on a model swap).
6. **Human review as the safety net.** Everything under the threshold goes to a reviewer; released
   reviews become labels; labels re-calibrate. Precision of the *combined* system (machine +
   review) is what the public trust page reports.
7. **Adversarial suites in CI.** Substitution suite at 100% catch; add fabricated-citation,
   altered-number, negation-flip, out-of-corpus and no-answer suites with the same release rule.
8. **Domain specifics.** Ontology extractor (scispaCy/UMLS) for entities the rules miss; retraction
   status and study-design tiers in ranking; recency weighting for guidelines.

## 3. What the product already enforces

- Hard vetoes before any score: fabricated citation, uncited critical claim, contradicted critical
  claim, entity substitution, numeric mismatch, out-of-scope → never delivered.
- Claim-level repair: unsupported non-critical claims are removed, not delivered.
- Separate families for generator, verifier and offline judge; no gold labels at runtime.
- Human review of blocked cases with re-verification of edited releases.
- Every answer carries versions and the calibration status; "uncalibrated" is shown until E4.

## 4. What is honest to say today

Nothing above 0% has been measured on real models. The architecture makes a 95–98% delivered
precision reachable at some coverage; the coverage is unknown until the baseline runs. The
thesis system released 64 answers of which 56% of yes/no answers were wrong; the production gate
is designed so that the same answers would have been withheld. That is a design claim, not a
measurement, until E4.

## 5. Token use (where the prompt budget goes)

| Item | Before | Now |
|---|---|---|
| JSON schema in the prompt | ~420 tokens pretty-printed | ~250 tokens compact |
| Evidence passages | whole passage per unit | question-relevant window (≤ 4 sentences, offsets kept); the verifier still sees the full passage |
| Repeated questions | new generation each time | cached by model + prompt + question + exact evidence spans (never stale across bundles) |
| Out-of-scope questions | retrieved and generated anyway | stopped before retrieval |
| Verifier pairs | unbounded | capped, cached per (model, premise, hypothesis) |
| Hosted spend | unbounded | per-request and per-day caps; abstention with `resource_limit` |

Measured savings per answer are reported in `evidence_fit.evidence_tokens_full` vs
`evidence_tokens_sent` and in the usage ledger once a real generator runs.
