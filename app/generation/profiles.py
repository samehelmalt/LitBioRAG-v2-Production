"""Per-model-family generation profiles: chat-template flags, decoding controls, stop strings,
reasoning-tag handling. The LitBioRAG experiments showed family-specific failures (DeepSeek-R1
distills repeating sentences and leaking <think> blocks; Qwen3 "thinking" tokens wasting the
budget), so the adapter must be model-aware. Profiles are configuration
(``configs/generation.yaml: profiles``), matched by model-id pattern; nothing here names a winner.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_PROFILES: list[dict[str, Any]] = [
    {
        "name": "qwen3",
        "match": r"(?i)qwen3",
        "chat_template_kwargs": {"enable_thinking": False},
        "decoding": {"repetition_penalty": 1.05, "no_repeat_ngram_size": 6},
        "stop_strings": ["<|im_end|>", "<|endoftext|>"],
        "reasoning_tags": ["think"],
    },
    {
        "name": "deepseek-r1-distill",
        "match": r"(?i)deepseek.*(r1|reason|distill)",
        "chat_template_kwargs": {},
        "decoding": {"repetition_penalty": 1.15, "no_repeat_ngram_size": 4},
        "stop_strings": ["<｜end▁of▁sentence｜>", "<|end_of_sentence|>"],
        "reasoning_tags": ["think"],
        "reasoning_budget_tokens": 512,
    },
    {
        "name": "llama-gemma-mistral",
        "match": r"(?i)llama|gemma|mistral|phi",
        "chat_template_kwargs": {},
        "decoding": {"repetition_penalty": 1.05, "no_repeat_ngram_size": 6},
        "stop_strings": [],
        "reasoning_tags": [],
    },
]


class GeneratorProfile(BaseModel):
    name: str = "generic"
    match: str = ".*"
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    decoding: dict[str, Any] = Field(default_factory=dict)  # transformers.generate kwargs
    stop_strings: list[str] = Field(default_factory=list)
    reasoning_tags: list[str] = Field(default_factory=list)  # <tag>…</tag> blocks stripped
    reasoning_budget_tokens: int | None = None  # extra tokens allowed for hidden reasoning
    json_mode: bool = True

    def strip_reasoning(self, text: str) -> tuple[str, str]:
        """Return (final_text, reasoning_text). Unclosed reasoning blocks are cut at the end."""
        reasoning: list[str] = []
        out = text
        for tag in self.reasoning_tags:
            pat = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
            reasoning.extend(m.group(1) for m in pat.finditer(out))
            out = pat.sub("", out)
            open_pat = re.compile(rf"<{tag}>.*\Z", re.DOTALL | re.IGNORECASE)
            m = open_pat.search(out)
            if m:
                reasoning.append(m.group(0))
                out = out[: m.start()]
        for s in self.stop_strings:
            out = out.split(s)[0]
        return out.strip(), "\n".join(r.strip() for r in reasoning)

    def max_new_tokens(self, max_tokens: int) -> int:
        return max_tokens + (self.reasoning_budget_tokens or 0)


def select_profile(model_id: str, profiles: list[dict[str, Any]] | None = None) -> GeneratorProfile:
    for raw in profiles if profiles is not None else DEFAULT_PROFILES:
        p = GeneratorProfile.model_validate(raw)
        if re.search(p.match, model_id):
            return p
    return GeneratorProfile()
