"""Generator protocol and adapters.

``Generator.generate`` takes chat messages and returns the raw model text; parsing happens in the
contract module. Adapters:

* ``TemplateGenerator`` — deterministic test double. Builds schema-valid JSON from the evidence
  by selecting sentences that overlap the question, so the whole pipeline runs in CI without
  weights. Knobs inject fabricated citations, altered numbers, swapped entities or flipped
  negation for adversarial tests of the verifier.
* ``HFCausalGenerator`` — local Hugging Face causal LM, optional 4-bit/8-bit (``ml`` extra).
* ``OpenAICompatibleGenerator`` — any /v1/chat/completions server (vLLM, Ollama, TGI, OpenAI).
* ``AnthropicGenerator`` — Anthropic Messages API (``anthropic`` package, optional).
Model choice is configuration (configs/models.yaml); none of these classes is named in code paths.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any, Protocol

from app.core.budget import HeuristicCounter, Usage
from app.evidence.selector import EvidenceUnit

Messages = Sequence[dict[str, str]]


class Generator(Protocol):
    name: str

    def generate(
        self,
        messages: Messages,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        evidence: Sequence[EvidenceUnit] | None = None,
    ) -> str: ...


# ---------------------------------------------------------------------- test double ------------
class TemplateGenerator:
    """Evidence-extractive JSON writer; deterministic; no model."""

    _tok = re.compile(r"[a-z0-9]+")
    _sent = re.compile(r"(?<=[.!?])\s+")
    _num = re.compile(r"(\d+(?:\.\d+)?)(\s*%)")
    _stop = {"the", "a", "an", "of", "in", "is", "are", "does", "do", "what", "which", "why", "how",
             "to", "and", "or", "for", "on", "with", "by", "that", "this", "it", "be"}  # fmt: skip

    def __init__(
        self,
        *,
        max_claims: int = 3,
        fabricate_citation: str | None = None,
        alter_numbers: bool = False,
        swap_entity: tuple[str, str] | None = None,
        flip_negation: bool = False,
        always_abstain: bool = False,
        emit_invalid_json: bool = False,
    ):
        self.name = "template:evidence-extractive"
        self.max_claims = max_claims
        self.fabricate_citation = fabricate_citation
        self.alter_numbers = alter_numbers
        self.swap_entity = swap_entity
        self.flip_negation = flip_negation
        self.always_abstain = always_abstain
        self.emit_invalid_json = emit_invalid_json

    def _question(self, messages: Messages) -> str:
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        m = re.search(r"Question:\s*(.+?)\n\s*\n", user, re.DOTALL)
        return m.group(1).strip() if m else user[:200]

    def _mutate(self, text: str) -> str:
        if self.alter_numbers:
            text = self._num.sub(lambda m: f"{float(m.group(1)) + 10:g}{m.group(2)}", text, count=1)
        if self.swap_entity:
            a, b = self.swap_entity
            text = text.replace(a, b)
        if self.flip_negation:
            text = (
                text.replace(" does not ", " does ")
                if " does not " in text
                else text.replace(" is required ", " is not required ")
            )
        return text

    def generate(
        self,
        messages: Messages,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        evidence: Sequence[EvidenceUnit] | None = None,
    ) -> str:
        if self.emit_invalid_json:
            return "I cannot format this as JSON, sorry."
        if self.always_abstain or not evidence:
            return json.dumps(
                {"answer": "", "abstained": True, "abstain_reason": "insufficient_evidence",
                 "claims": [], "conflicts": []}  # fmt: skip
            )
        q = set(self._tok.findall(self._question(messages).lower())) - self._stop
        scored: list[tuple[int, int, str, str]] = []
        for rank, u in enumerate(evidence):
            for s in self._sent.split(u.text):
                s = s.strip()
                if len(s) < 15:
                    continue
                overlap = len(q & set(self._tok.findall(s.lower())))
                if overlap:
                    scored.append((-overlap, rank, s, u.citation_id))
        scored.sort()
        claims = []
        for i, (_, _, s, cit) in enumerate(scored[: self.max_claims]):
            text = self._mutate(s)
            cits = [self.fabricate_citation] if (self.fabricate_citation and i == 0) else [cit]
            claims.append({"text": text, "citations": cits, "critical": i == 0})
        if not claims:
            return json.dumps(
                {"answer": "", "abstained": True, "abstain_reason": "insufficient_evidence",
                 "claims": [], "conflicts": []}  # fmt: skip
            )
        return json.dumps(
            {"answer": " ".join(c["text"] for c in claims), "abstained": False,
             "claims": claims, "conflicts": []}  # fmt: skip
        )


# ---------------------------------------------------------------------- local HF ---------------
class HFCausalGenerator:
    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        quantization: str | None = None,
        device: str | None = None,
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf_causal:{model_id}" + (f"@{revision}" if revision else "")
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        kwargs: dict[str, Any] = {"revision": revision}
        if quantization in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit",
                load_in_8bit=quantization == "8bit",
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = device or "auto"
        else:
            kwargs["torch_dtype"] = getattr(torch, dtype)
            kwargs["device_map"] = device or "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        self.last_usage: Usage | None = None

    def generate(self, messages, *, max_tokens=1024, temperature=0.0, evidence=None) -> str:
        t0 = time.perf_counter()
        ids = self.tokenizer.apply_chat_template(
            list(messages), add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        gen: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            gen.update(do_sample=False)
        with self._torch.inference_mode():
            out = self.model.generate(ids, **gen)
        new = out[0][ids.shape[1] :]
        self.last_usage = Usage(model=self.name, prompt_tokens=int(ids.shape[1]),
                                completion_tokens=int(new.shape[0]), exact=True,
                                latency_ms=(time.perf_counter() - t0) * 1000)  # fmt: skip
        return self.tokenizer.decode(new, skip_special_tokens=True)


# ---------------------------------------------------------------------- OpenAI-compatible -------
class OpenAICompatibleGenerator:
    def __init__(
        self,
        model_id: str,
        *,
        endpoint: str = "http://localhost:8000/v1",
        api_key_env: str | None = None,
        timeout: int = 60,
        json_mode: bool = True,
        max_retries: int = 3,
        backoff_s: float = 2.0,
    ):
        self.name = f"openai_compatible:{model_id}"
        self.model_id = model_id
        self.endpoint = endpoint.rstrip("/")
        self.api_key_env = api_key_env
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.timeout = timeout
        self.json_mode = json_mode
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.last_usage: Usage | None = None
        self._counter = HeuristicCounter()

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.endpoint}/chat/completions", json.dumps(body).encode(), headers=headers
        )
        attempt = 0
        while True:
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                retryable = e.code in (408, 409, 425, 429, 500, 502, 503, 504)
                if not retryable or attempt >= self.max_retries:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt >= self.max_retries:
                    raise
            time.sleep(self.backoff_s * (2**attempt))
            attempt += 1

    def generate(self, messages, *, max_tokens=1024, temperature=0.0, evidence=None) -> str:
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        data = self._post(body)
        text = data["choices"][0]["message"]["content"] or ""
        u = data.get("usage") or {}
        self.last_usage = Usage(
            model=self.name,
            prompt_tokens=int(u.get("prompt_tokens") or sum(
                self._counter.count(m.get("content", "")) for m in messages)),
            completion_tokens=int(u.get("completion_tokens") or self._counter.count(text)),
            exact=bool(u),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )  # fmt: skip
        return text


# ---------------------------------------------------------------------- Anthropic --------------
class AnthropicGenerator:
    def __init__(self, model_id: str, *, api_key_env: str = "ANTHROPIC_API_KEY"):
        import anthropic  # optional dependency

        self.name = f"anthropic:{model_id}"
        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=os.environ.get(api_key_env))
        self.last_usage: Usage | None = None

    def generate(self, messages, *, max_tokens=1024, temperature=0.0, evidence=None) -> str:
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        rest = [m for m in messages if m.get("role") != "system"]
        t0 = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model_id,
            system=system,
            messages=rest,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        u = getattr(resp, "usage", None)
        self.last_usage = Usage(
            model=self.name,
            prompt_tokens=int(getattr(u, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(u, "output_tokens", 0) or 0),
            exact=u is not None,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return "".join(getattr(b, "text", "") for b in resp.content)
