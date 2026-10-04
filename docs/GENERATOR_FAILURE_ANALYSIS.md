# Generator Failure Analysis

**Status: NOT YET MEASURED on the production system.** Seeded from thesis observations; populated in Step 6 once candidate generators run under the production contract on the harness.

## Failure taxonomy

| Code | Failure | Detection in the verification pipeline |
|---|---|---|
| HAL | hallucination (claim with no evidence counterpart) | no passage with entailment above threshold |
| UNS | unsupported claim (related evidence exists but does not entail) | evidence mapped, entailment below threshold, no contradiction |
| CON | contradiction | contradiction above threshold |
| GEN | overgeneralization (population, dose, setting widened) | entity/constraint mismatch between claim and evidence |
| CAUS | causal overclaim from observational evidence | study-type metadata vs causal verb in claim |
| ENT | entity substitution (MAP3K3 → MAP3K7, drug analogue) | entity consistency check |
| NUM | numeric hallucination or alteration | unit-aware numeric check |
| CIT | citation hallucination (PMID/DOI not in retrieved set, or does not support the claim) | citation resolution |
| CTX | context confusion (right fact, wrong document attributed) | claim supported by a different passage than cited |
| ABS- | failure to abstain when evidence is insufficient | retrieval sufficiency low ∧ answer given |
| ABS+ | unnecessary abstention | evidence sufficient ∧ abstained |
| FMT | contract violation (schema, polarity line, missing citations) | parser |

## Seed observations from the thesis run (★ recomputed from the CSVs)

- Yes/no accuracy at S3a: Llama-3.2-3B 0.76, DeepSeek-R1-Distill-Llama-3B 0.71, DeepSeek-1.5B 0.52, Qwen2.5-3B 0.24, Qwen2-1.5B 0.24, LitRag 0.10; NoRAG Qwen2.5-3B 0.29. Retrieval lowered the structured models' accuracy → candidate causes FMT and CTX (format-forcing prompt, whole-abstract context).
- SBE (S5→S6) lowered NLI faithfulness 0.273 → 0.138 and Llama-3B yes/no 0.76 → 0.33.
- Prompts contained "MUST NOT abstain" when evidence was retrieved → ABS- by design.
- Mean unsupported-claim rate among PASS answers 0.44 → UNS/HAL survive the gate.
- Format compliance (thesis column): Qwen2.5-3B 1.00, Qwen-1.5B 0.99, DeepSeek-1.5B 0.88, Llama-3B 0.85, DeepSeek-3B 0.71, LitRag 0.67.

## Planned measurements (Step 6)

For each candidate generator under the production contract, on the validation split: claim count per answer, share of claims by verdict, failure code distribution, abstention precision/recall against proxy labels, citation validity rate, numeric and entity fidelity, yes/no accuracy, VRAM at load, tokens/s, and the delta between bf16 and 4-bit where both fit.
