# Model registry

Every model role is resolved from `configs/models.yaml`; code never names a model.

| Role | Used by | Kinds |
|---|---|---|
| `dense_encoder` | retrieval (pinned by the bundle) | `sentence_transformers`, `hash` (test) |
| `reranker` | single-stage reranking | `hf_seq_cls`, `lexical` (test) |
| `generator` | answer generation | `hf_causal`, `openai_compatible`, `anthropic`, `template` (test) |
| `verifier` | claim entailment / contradiction | `hf_seq_cls`, `minicheck`, `overlap` (test) |
| `entity_extractor` | entity and relation grounding | `regex`, `scispacy` |
| `offline_judge` | evaluation only, never at runtime | `openai_compatible`, `anthropic`, `hf_causal` |

## Swap a model (no code change)

Edit the profile in `configs/models.yaml`:

```yaml
generator:
  kind: openai_compatible          # any /v1/chat/completions server: vLLM, Ollama, TGI, OpenAI
  model_id: Qwen/Qwen3-14B
  endpoint: http://vllm:8000/v1
  api_key_env: VLLM_API_KEY
```

or override one role for one process:

```bash
export LITBIORAG_MODEL__GENERATOR="openai_compatible:Qwen/Qwen3-14B?endpoint=http://vllm:8000/v1"
export LITBIORAG_MODEL_PROFILE=default
```

Spec string grammar: `kind:model_id[@revision][?key=value&...]`; recognised keys `quantization`, `device`, `endpoint`, `api_key_env`, `max_tokens`, `family`; anything else lands in `params` for the adapter.

## Rules the loader enforces

1. **Independence.** `generator`, `verifier` and `offline_judge` must come from different model families (the thesis reused one NLI family across selection, gating and evaluation). Test doubles are exempt.
2. **Calibration follows the model.** `thresholds()` are returned only when `calibration.for_model` equals the role's `model_id`. Swapping a model empties its thresholds until the calibration job (Step 8) writes new ones. Profiles with `require_calibrated: false` (test, thesis-baseline) return `{}` instead of refusing.
3. **The bundle pins the encoder.** `DenseIndex` refuses an encoder whose name differs from `dense/meta.json`; changing the encoder means re-embedding on the build host, not editing YAML alone.
4. **Unknown kind → explicit error** listing the registered adapters; nothing falls back silently.

## Add a new kind (code change, once)

One factory in `models/registry/adapters.py` (or in the package that owns the role):

```python
@adapter("my_kind", "generator")
def _my_generator(spec: ModelSpec):
    return MyGenerator(spec.model_id, **spec.params)
```

After that, every model of that kind is a YAML entry.
