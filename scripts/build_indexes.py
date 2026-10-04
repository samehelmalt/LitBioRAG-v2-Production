"""Build an artifact bundle from JSONL documents (BUILD host).

    python -m scripts.build_indexes --docs "data/pubmed/jsonl/*.jsonl" "data/preprints/*.jsonl" \
        --out artifacts/pubmed2015plus_v1 --corpus-id pubmed2015plus_v1 \
        --encoder hf:ncbi/MedCPT-Article-Encoder --batch-size 128 --shard-size 100000

Stages, each resumable (a finished stage leaves a marker and is skipped on re-run):
  1. store   JSONL documents -> metadata.sqlite (documents + passages)
  2. bm25    passages -> bm25/ (Tantivy)
  3. embed   passages -> dense/shards/NNNNN.npy (one file per shard; a crash resumes at the
             first missing shard)
  4. dense   shards -> dense/index.faiss + ids.npy
  5. pack    manifest.json with sha256 per file
Afterwards copy the bundle directory to Google Drive and run ``scripts.artifacts fetch`` on the
serving host.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.documents import Document  # noqa: E402
from app.retrieval.bm25 import BM25Index  # noqa: E402
from app.retrieval.dense import build_dense_index  # noqa: E402
from app.retrieval.encoders import make_encoder  # noqa: E402
from app.retrieval.passages import split_document  # noqa: E402
from app.retrieval.store import MetadataStore  # noqa: E402
from scripts.artifacts import pack  # noqa: E402


def _marker(out: Path, stage: str) -> Path:
    return out / f".{stage}.done"


def _log(stage: str, **kw) -> None:
    print(json.dumps({"stage": stage, "t": round(time.time(), 1), **kw}), flush=True)


def iter_docs(patterns: list[str]):
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield Document.model_validate_json(line)


def stage_store(patterns: list[str], out: Path, *, target_words: int, max_words: int) -> dict:
    db = out / "metadata.sqlite"
    if _marker(out, "store").exists():
        return {"skipped": True}
    if db.exists():
        db.unlink()  # a half-written store is rebuilt from scratch
    store = MetadataStore(db)
    n_docs = n_pass = 0
    doc_buf, pas_buf = [], []
    for doc in iter_docs(patterns):
        doc_buf.append(doc)
        pas_buf.extend(split_document(doc, target_words=target_words, max_words=max_words))
        if len(doc_buf) >= 5000:
            n_docs += store.add_documents(doc_buf)
            n_pass += store.add_passages(pas_buf)
            doc_buf, pas_buf = [], []
    n_docs += store.add_documents(doc_buf)
    n_pass += store.add_passages(pas_buf)
    store.set_meta("passage_target_words", str(target_words))
    store.set_meta("passage_max_words", str(max_words))
    store.close()
    _marker(out, "store").touch()
    return {"documents": n_docs, "passages": n_pass}


def stage_bm25(out: Path) -> dict:
    if _marker(out, "bm25").exists():
        return {"skipped": True}
    store = MetadataStore(out / "metadata.sqlite", read_only=True)
    meta = {
        r["doc_id"]: (r["year"], r["source_type"])
        for r in store.conn.execute("SELECT doc_id, year, source_type FROM documents")
    }
    idx = BM25Index(out / "bm25")
    n = idx.add(store.iter_passages(), meta)
    store.close()
    _marker(out, "bm25").touch()
    return {"passages": n}


def stage_embed(out: Path, encoder_spec: str, *, shard_size: int, batch_size: int) -> dict:
    shards_dir = out / "dense" / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    store = MetadataStore(out / "metadata.sqlite", read_only=True)
    n_total = store.counts()["passages"]
    n_shards = (n_total + shard_size - 1) // shard_size
    ids_path = out / "dense" / "ids.npy"
    if _marker(out, "embed").exists():
        store.close()
        return {"skipped": True, "shards": n_shards}
    encoder = make_encoder(encoder_spec)
    ids: list[str] = []
    buf_ids: list[str] = []
    buf_txt: list[str] = []
    shard_no = 0

    def flush() -> None:
        nonlocal shard_no
        path = shards_dir / f"{shard_no:05d}.npy"
        if not path.exists():
            vecs = []
            for i in range(0, len(buf_txt), batch_size):
                vecs.append(encoder.encode(buf_txt[i : i + batch_size], batch_size=batch_size))
            tmp = path.with_suffix(".part.npy")
            np.save(tmp, np.concatenate(vecs).astype("float32"))
            tmp.replace(path)
            _log("embed", shard=shard_no, n=len(buf_txt))
        shard_no += 1

    for p in store.iter_passages():
        ids.append(p.passage_id)
        buf_ids.append(p.passage_id)
        buf_txt.append(p.text)
        if len(buf_txt) >= shard_size:
            flush()
            buf_ids, buf_txt = [], []
    if buf_txt:
        flush()
    store.close()
    np.save(ids_path, np.asarray(ids, dtype=object), allow_pickle=True)
    (out / "dense" / "encoder.json").write_text(
        json.dumps({"encoder": encoder.name, "dim": encoder.dim, "shards": shard_no})
    )
    _marker(out, "embed").touch()
    return {"passages": len(ids), "shards": shard_no, "encoder": encoder.name}


def stage_dense(out: Path) -> dict:
    if _marker(out, "dense").exists():
        return {"skipped": True}
    info = json.loads((out / "dense" / "encoder.json").read_text())
    ids = list(np.load(out / "dense" / "ids.npy", allow_pickle=True))
    shard_files = sorted((out / "dense" / "shards").glob("[0-9]*.npy"))
    meta = build_dense_index(
        (np.load(f) for f in shard_files),
        ids,
        out / "dense",
        encoder_name=info["encoder"],
        dim=info["dim"],
    )
    _marker(out, "dense").touch()
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--docs", nargs="+", required=True, help="glob(s) of JSONL document files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corpus-id", required=True)
    ap.add_argument("--encoder", default="hash:256", help="hash:<dim> or hf:<model_id>")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--shard-size", type=int, default=100_000)
    ap.add_argument("--target-words", type=int, default=200)
    ap.add_argument("--max-words", type=int, default=300)
    ap.add_argument("--keep-shards", action="store_true", help="keep dense/shards after indexing")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    _log("store", **stage_store(a.docs, a.out, target_words=a.target_words, max_words=a.max_words))
    _log("bm25", **stage_bm25(a.out))
    _log("embed", **stage_embed(a.out, a.encoder, shard_size=a.shard_size, batch_size=a.batch_size))
    _log("dense", **stage_dense(a.out))
    if not a.keep_shards:
        for f in (a.out / "dense" / "shards").glob("*.npy"):
            f.unlink()
    store = MetadataStore(a.out / "metadata.sqlite", read_only=True)
    counts = store.counts()
    store.close()
    m = pack(
        a.out,
        corpus_id=a.corpus_id,
        extra={
            "counts": counts,
            "encoder": json.loads((a.out / "dense" / "encoder.json").read_text()),
            "dense": json.loads((a.out / "dense" / "meta.json").read_text()),
            "passage_params": {"target_words": a.target_words, "max_words": a.max_words},
        },
    )
    _log("pack", files=len(m["files"]), total_mb=round(m["total_bytes"] / 1e6, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
