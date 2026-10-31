"""Verifier protocol and adapters.

A verifier scores (premise, hypothesis) pairs and returns entail / neutral / contradict
probabilities. ``HFNLIVerifier`` wraps any NLI sequence-classification model; long premises are
chunked at the token level and the claim is scored against every chunk, taking the **maximum
entailment and the maximum contradiction across chunks** — not the most-confident label, which in
the thesis let a confident NEUTRAL hide a weaker ENTAILMENT (PRODUCTION_AUDIT.md §4).
``OverlapVerifier`` is a deterministic, negation-aware stand-in for CI. The verifier must come
from a different model family than the generator; the registry enforces it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel

Pair = tuple[str, str]  # (premise, hypothesis)


class NLIScores(BaseModel):
    entail: float
    neutral: float
    contradict: float


class Verifier(Protocol):
    name: str

    def score_pairs(self, pairs: Sequence[Pair]) -> list[NLIScores]: ...


# ---------------------------------------------------------------- deterministic double ----------
_NEG = re.compile(
    r"\b(not|no|never|without|neither|nor|cannot|lacks?|absen(?:t|ce)|fail(?:ed|s)? to|"
    r"did not|does not|was not|were not|is not|are not|no significant)\b"
)
_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "in", "is", "are", "was", "were", "to", "and", "or", "for",
         "on", "with", "by", "that", "this", "it", "be", "as", "at", "from", "we", "show",
         "showed", "study", "studies", "also", "known"}  # fmt: skip


def _content(text: str) -> set[str]:
    return {t for t in _TOK.findall(text.lower()) if t not in _STOP and len(t) > 1}


def _negated(text: str) -> bool:
    return bool(_NEG.search(text.lower()))


class OverlapVerifier:
    """Entails when most content tokens of the hypothesis occur in the premise with the same
    negation polarity; contradicts when the overlap is high but polarity differs; else neutral."""

    name = "overlap:negation-aware"

    def __init__(self, min_overlap: float = 0.6):
        self.min_overlap = min_overlap

    def score_one(self, premise: str, hypothesis: str) -> NLIScores:
        h = _content(hypothesis)
        if not h:
            return NLIScores(entail=0.0, neutral=1.0, contradict=0.0)
        p = _content(premise)
        overlap = len(h & p) / len(h)
        if overlap < self.min_overlap:
            return NLIScores(entail=overlap * 0.5, neutral=1 - overlap * 0.5, contradict=0.0)
        # polarity: compare negation in the premise sentence that best matches the hypothesis
        sents = re.split(r"(?<=[.!?])\s+", premise)
        best = max(sents, key=lambda s: len(h & _content(s)), default=premise)
        if _negated(best) != _negated(hypothesis):
            return NLIScores(entail=0.05, neutral=0.05, contradict=0.90)
        return NLIScores(entail=min(0.99, 0.5 + overlap / 2), neutral=0.0, contradict=0.02)

    def score_pairs(self, pairs: Sequence[Pair]) -> list[NLIScores]:
        return [self.score_one(p, h) for p, h in pairs]


# ---------------------------------------------------------------- Hugging Face NLI --------------
class HFNLIVerifier:
    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int | None = None,
        dtype: str = "float16",
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = f"hf_seq_cls:{model_id}" + (f"@{revision}" if revision else "")
        self._torch = torch
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        torch_dtype = getattr(torch, dtype) if self.device != "cpu" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, torch_dtype=torch_dtype, revision=revision
        )
        self.model.to(self.device).eval()
        model_max = getattr(self.tokenizer, "model_max_length", 512) or 512
        self.max_length = min(max_length or model_max, model_max, 4096)
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.idx = {}
        for i, lab in id2label.items():
            if "entail" in lab:
                self.idx["entail"] = i
            elif "contra" in lab:
                self.idx["contradict"] = i
            elif "neutral" in lab:
                self.idx["neutral"] = i
        if set(self.idx) != {"entail", "neutral", "contradict"}:
            raise ValueError(f"{model_id}: cannot map labels {id2label} to NLI classes")

    def _chunks(self, premise: str, hypothesis: str) -> list[str]:
        hyp_len = len(self.tokenizer(hypothesis, add_special_tokens=False)["input_ids"])
        window = max(64, self.max_length - hyp_len - 8)
        ids = self.tokenizer(premise, add_special_tokens=False)["input_ids"]
        if len(ids) <= window:
            return [premise]
        stride = window // 2
        return [
            self.tokenizer.decode(ids[i : i + window], skip_special_tokens=True)
            for i in range(0, len(ids), stride)
            if ids[i : i + window]
        ]

    def score_pairs(self, pairs: Sequence[Pair]) -> list[NLIScores]:
        expanded: list[tuple[int, str, str]] = []
        for k, (p, h) in enumerate(pairs):
            for c in self._chunks(p, h):
                expanded.append((k, c, h))
        best: dict[int, NLIScores] = {}
        with self._torch.inference_mode():
            for i in range(0, len(expanded), self.batch_size):
                batch = expanded[i : i + self.batch_size]
                enc = self.tokenizer(
                    [b[1] for b in batch], [b[2] for b in batch],
                    padding=True, truncation="only_first", max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)  # fmt: skip
                probs = self.model(**enc).logits.float().softmax(-1).cpu().tolist()
                for (k, _, _), pr in zip(batch, probs, strict=True):
                    s = NLIScores(entail=pr[self.idx["entail"]], neutral=pr[self.idx["neutral"]],
                                  contradict=pr[self.idx["contradict"]])  # fmt: skip
                    cur = best.get(k)
                    if cur is None:
                        best[k] = s
                    else:  # max entailment AND max contradiction across chunks
                        best[k] = NLIScores(
                            entail=max(cur.entail, s.entail),
                            neutral=min(cur.neutral, s.neutral),
                            contradict=max(cur.contradict, s.contradict),
                        )
        return [best[k] for k in range(len(pairs))]
