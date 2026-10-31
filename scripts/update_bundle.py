"""Incrementally add documents (JSONL) to an existing bundle: store + BM25 + dense, then re-pack.

    python -m scripts.update_bundle --bundle data/artifacts/<id> --docs new_pubmed_*.jsonl \
        --encoder hf:ncbi/MedCPT-Article-Encoder

Used for PubMed daily update files and new preprint fetches ("update the new added") without a
full rebuild. The dense index must have been built with the same encoder (checked). IVF-PQ and
Flat indexes both accept ``add`` after training, so no re-training is needed. Documents already
present are replaced (same doc id) — their old passages are removed from the store first; stale
vectors for replaced passages are tolerated (same passage ids are re-added and win on rerank)
until the next full rebuild, which is reported in the manifest as ``stale_vectors``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app.retrieval.bm25 import BM25Index  # noqa: E402
from app.retrieval.encoders import make_encoder  # noqa: E402
from app.retrieval.passages import split_document  # noqa: E402
from app.retrieval.store import MetadataStore  # noqa: E402
from scripts.artifacts import pack  # noqa: E402
from scripts.build_indexes import iter_docs  # noqa: E402


def update_bundle(bundle: Path, patterns: list[str], encoder_spec: str, *,
                  batch_size: int = 64, target_words: int | None = None,
                  max_words: int | None = None) -> dict:  # fmt: skip
    import faiss

    manifest = json.loads((bundle / "manifest.json").read_text())
    dense_meta = json.loads((bundle / "dense" / "meta.json").read_text())
    encoder = make_encoder(encoder_spec)
    if dense_meta["encoder"] != encoder.name:
        raise ValueError(f"bundle dense index uses {dense_meta['encoder']!r}, got {encoder.name!r}")
    store = MetadataStore(bundle / "metadata.sqlite")
    tw = target_words or int(store.get_meta("passage_target_words") or 200)
    mw = max_words or int(store.get_meta("passage_max_words") or 300)
    docs = list(iter_docs(patterns))
    if not docs:
        return {"added_documents": 0}
    # replace existing versions of the same documents
    existing = [d.doc_id for d in docs if store.get_document(d.doc_id) is not None]
    stale = 0
    for did in existing:
        stale += len(store.passages_of(did))
    store.delete_documents(existing)
    passages = []
    for d in docs:
        passages.extend(split_document(d, target_words=tw, max_words=mw))
    store.add_documents(docs)
    store.add_passages(passages)
    meta = {d.doc_id: (d.year, d.source_type.value, d.course_id) for d in docs}
    BM25Index(bundle / "bm25").add(passages, meta)
    # dense: encode new passages and append
    index = faiss.read_index(str(bundle / "dense" / "index.faiss"))
    ids = list(np.load(bundle / "dense" / "ids.npy", allow_pickle=True))
    texts = [p.text for p in passages]
    vecs = []
    for i in range(0, len(texts), batch_size):
        vecs.append(encoder.encode(texts[i : i + batch_size], batch_size=batch_size))
    v = np.ascontiguousarray(np.concatenate(vecs), dtype="float32")
    index.add(v)
    ids.extend(p.passage_id for p in passages)
    faiss.write_index(index, str(bundle / "dense" / "index.faiss"))
    np.save(bundle / "dense" / "ids.npy", np.asarray(ids, dtype=object), allow_pickle=True)
    dense_meta["n"] = len(ids)
    dense_meta["stale_vectors"] = int(dense_meta.get("stale_vectors", 0)) + stale
    (bundle / "dense" / "meta.json").write_text(json.dumps(dense_meta, indent=1))
    counts = store.counts()
    store.close()
    extra = {k: v for k, v in manifest.items() if k not in ("files", "total_bytes", "created_at")}
    extra["counts"] = counts
    extra["dense"] = dense_meta
    updates = list(extra.get("updates", []))
    updates.append({"documents": len(docs), "passages": len(passages), "replaced": len(existing)})
    extra["updates"] = updates
    pack(bundle, corpus_id=manifest["corpus_id"], extra=extra)
    return {"added_documents": len(docs), "added_passages": len(passages),
            "replaced": len(existing), "stale_vectors": dense_meta["stale_vectors"]}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args(argv)
    print(json.dumps(update_bundle(a.bundle, a.docs, a.encoder, batch_size=a.batch_size)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
