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
        revision: str | None = None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = f"hf_seq_cls:{model_id}" + (f"@{revision}" if revision else "")
        self.batch_size = batch_size
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        torch_dtype = getattr(torch, dtype) if self.device != "cpu" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, torch_dtype=torch_dtype, revision=revision
        )
        self.model.to(self.device).eval()
        model_max = getattr(self.tokenizer, "model_max_length", 512) or 512
        self.max_length = min(max_length or model_max, model_max, 8192)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Relevance score per (query, passage) pair; pairs may come from different queries, which
        is what lets ``app.core.batcher`` share one forward pass across concurrent requests."""
        scores: list[float] = []
        with self._torch.inference_mode():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                enc = self.tokenizer(
                    [q for q, _ in batch],
                    [t for _, t in batch],
                    padding=True,
                    truncation="only_second",  # never cut the query
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                # single-logit relevance heads (MedCPT, BGE) vs 2-class heads
                col = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
                scores.extend(col.float().cpu().tolist())
        return scores

    def rerank(self, query: str, candidates: Sequence[Candidate], top_n: int) -> list[RerankHit]:
        if not candidates:
            return []
        scores = self.score_pairs([(query, c.text) for c in candidates])
        hits = [RerankHit(c.passage_id, s) for c, s in zip(candidates, scores, strict=True)]
        hits.sort(key=lambda h: (-h.score, h.passage_id))
        return hits[:top_n]


def make_reranker(spec: str):
    """Build a reranker from a registry spec string; 'none' -> None.

    'lexical' is shorthand for 'lexical:overlap', 'hf:<id>' for 'hf_seq_cls:<id>'. Model choice
    lives in configs/models.yaml; this helper exists for CLIs that take a spec on the command line.
    """
    from models.registry import ModelSpec
    from models.registry.loader import build_spec

    if spec in ("none", "", None):
        return None
    if spec == "lexical":
        spec = "lexical:overlap"
    elif spec.startswith("hf:"):
        spec = "hf_seq_cls:" + spec[3:]
    return build_spec(ModelSpec.parse(spec, "reranker"))
