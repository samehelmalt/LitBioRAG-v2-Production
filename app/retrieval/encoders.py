"""Dense encoders behind one small protocol.

``HashEncoder`` is a deterministic, dependency-free encoder used by tests and CI so the whole
index pipeline runs without model weights. ``SentenceTransformerEncoder`` wraps any
sentence-transformers model on the build host (lazy import; needs the ``ml`` extra).
Model choice is made by the benchmark in Step 5, not here.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized vectors."""
        ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class HashEncoder:
    """Bag-of-hashed-tokens vectors. Lexical only; adequate to exercise the pipeline."""

    _tok = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 256):
        self.name = f"hash:{dim}"
        self.dim = dim

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in self._tok.findall(t.lower()):
                h = int(hashlib.blake2b(tok.encode(), digest_size=4).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return l2_normalize(out)


class SentenceTransformerEncoder:
    def __init__(self, model_id: str, device: str | None = None, max_seq_length: int | None = None):
        from sentence_transformers import SentenceTransformer  # lazy: ml extra

        self._model = SentenceTransformer(model_id, device=device)
        if max_seq_length:
            self._model.max_seq_length = max_seq_length
        self.name = f"hf:{model_id}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        vecs = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")


def make_encoder(spec: str) -> Encoder:
    """'hash:<dim>' or 'hf:<model_id>'."""
    kind, _, arg = spec.partition(":")
    if kind == "hash":
        return HashEncoder(int(arg or 256))
    if kind == "hf":
        if not arg:
            raise ValueError("hf: needs a model id")
        return SentenceTransformerEncoder(arg)
    raise ValueError(f"unknown encoder spec {spec!r} (use hash:<dim> or hf:<model_id>)")
