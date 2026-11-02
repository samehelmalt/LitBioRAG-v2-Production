"""Dense encoders behind one small protocol.

``HashEncoder`` is a deterministic, dependency-free encoder used by tests and CI so the whole
index pipeline runs without model weights. ``SentenceTransformerEncoder`` wraps any
sentence-transformers model on the build host (lazy import; needs the ``ml`` extra).

Paired encoders: some biomedical retrievers are *asymmetric* — MedCPT has an Article encoder for
passages and a separate Query encoder for questions. Using the article tower for queries (what the
thesis did) silently degrades retrieval. ``Encoder.encode_queries`` exists for that reason: the
index is built with ``encode`` and searched with ``encode_queries``; a symmetric model simply uses
the same weights for both. The pairing is part of the encoder's name, so a bundle built with one
pairing refuses an encoder with another.
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
        """Return an (n, dim) float32 array of L2-normalized passage vectors."""
        ...

    def encode_queries(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Query-side vectors (same space as ``encode``). Symmetric encoders delegate."""
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
        self.query_calls = 0  # tests assert the query tower is the one used for searches

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in self._tok.findall(t.lower()):
                h = int(hashlib.blake2b(tok.encode(), digest_size=4).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return l2_normalize(out)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        self.query_calls += len(texts)
        return self.encode(texts, batch_size=batch_size)


class SentenceTransformerEncoder:
    """Passage tower ``model_id``; optional separate query tower ``query_model_id`` (MedCPT)."""

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        max_seq_length: int | None = None,
        revision: str | None = None,
        query_model_id: str | None = None,
        query_max_seq_length: int | None = None,
    ):
        from sentence_transformers import SentenceTransformer  # lazy: ml extra

        self._model = SentenceTransformer(model_id, device=device, revision=revision)
        if max_seq_length:
            self._model.max_seq_length = max_seq_length
        self._query_model = None
        if query_model_id and query_model_id != model_id:
            self._query_model = SentenceTransformer(query_model_id, device=device)
            self._query_model.max_seq_length = query_max_seq_length or 64
        self.model_id = model_id
        self.query_model_id = query_model_id if self._query_model is not None else model_id
        self.name = f"hf:{model_id}" + (f"?query={query_model_id}" if self._query_model else "")
        self.dim = int(self._model.get_sentence_embedding_dimension())
        if self._query_model is not None:
            qdim = int(self._query_model.get_sentence_embedding_dimension())
            if qdim != self.dim:
                raise ValueError(f"query tower dim {qdim} != article tower dim {self.dim}")

    def _run(self, model, texts: Sequence[str], batch_size: int) -> np.ndarray:
        vecs = model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return self._run(self._model, texts, batch_size)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return self._run(self._query_model or self._model, texts, batch_size)


# Known asymmetric families: article id -> query id. A profile that names the article tower
# without its query tower is refused by the registry (models/registry/loader.py).
PAIRED_QUERY_ENCODERS: dict[str, str] = {
    "ncbi/MedCPT-Article-Encoder": "ncbi/MedCPT-Query-Encoder",
}


def encode_queries(encoder: Encoder, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    """Use the query tower when the encoder has one (older wrappers may not)."""
    fn = getattr(encoder, "encode_queries", None)
    return fn(texts, batch_size=batch_size) if fn else encoder.encode(texts, batch_size=batch_size)


def make_encoder(spec: str) -> Encoder:
    """Build an encoder from a registry spec string.

    'hash:<dim>', 'sentence_transformers:<id>[?query=<query_id>]'; 'hf:<id>' is shorthand for
    'sentence_transformers:<id>'. Model choice lives in configs/models.yaml; this helper exists for
    CLIs that take a spec on the command line.
    """
    from models.registry import ModelSpec
    from models.registry.loader import build_spec

    if spec.startswith("hf:"):
        spec = "sentence_transformers:" + spec[3:]
    return build_spec(ModelSpec.parse(spec, "dense_encoder"))
