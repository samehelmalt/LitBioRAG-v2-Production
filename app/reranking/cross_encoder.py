"""Cross-encoder reranker over the full candidate pool at full token length.

Works with any Hugging Face sequence-classification cross-encoder (MedCPT-Cross-Encoder,
BGE-reranker, ms-marco MiniLM, ...). Truncation is done by the tokenizer at the model's maximum
sequence length (tokens), never by slicing characters: the thesis cut every passage to 512
characters (~80 words) before scoring (PRODUCTION_AUDIT.md §2), which is one of the retrieval
defects this module exists to remove. Lazy imports: needs the ``ml`` extra.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.reranking.base import Candidate, RerankHit


class CrossEncoderReranker:
    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int | None = None,
        dtype: str = "float16",
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = f"hf:{model_id}"
        self.batch_size = batch_size
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        torch_dtype = getattr(torch, dtype) if self.device != "cpu" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, torch_dtype=torch_dtype
        )
        self.model.to(self.device).eval()
        model_max = getattr(self.tokenizer, "model_max_length", 512) or 512
        self.max_length = min(max_length or model_max, model_max, 8192)

    def rerank(self, query: str, candidates: Sequence[Candidate], top_n: int) -> list[RerankHit]:
        if not candidates:
            return []
        scores: list[float] = []
        with self._torch.inference_mode():
            for i in range(0, len(candidates), self.batch_size):
                batch = candidates[i : i + self.batch_size]
                enc = self.tokenizer(
                    [query] * len(batch),
                    [c.text for c in batch],
                    padding=True,
                    truncation="only_second",  # never cut the query
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                # single-logit relevance heads (MedCPT, BGE) vs 2-class heads
                col = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
                scores.extend(col.float().cpu().tolist())
        hits = [RerankHit(c.passage_id, s) for c, s in zip(candidates, scores, strict=True)]
        hits.sort(key=lambda h: (-h.score, h.passage_id))
        return hits[:top_n]


def make_reranker(spec: str):
    """Resolve 'none' | 'lexical' | 'hf:<model_id>' to a reranker instance (None for 'none')."""
    if spec in ("none", "", None):
        return None
    if spec == "lexical":
        from app.reranking.base import LexicalOverlapReranker

        return LexicalOverlapReranker()
    kind, _, arg = spec.partition(":")
    if kind == "hf" and arg:
        return CrossEncoderReranker(arg)
    raise ValueError(f"unknown reranker spec {spec!r} (use none | lexical | hf:<model_id>)")
