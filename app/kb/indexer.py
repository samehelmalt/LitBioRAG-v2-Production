"""Index uploaded files into a per-server *uploads bundle* (store + BM25 + dense), incrementally.

Layout (``data/kb/``): ``files/<kb_id>/<file_id><ext>`` raw uploads, ``index/metadata.sqlite``,
``index/bm25``, ``index/dense``. Documents carry ``file`` provenance and ``course_id = kb_id``,
so the existing entitlement filter restricts retrieval to the caller's knowledge bases. BM25 is
appended to in place; the dense index is rebuilt from the store's passages after each change
(small enough for uploads; the main corpus stays a build-host job).
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from app.core.documents import Document
from app.kb.adapters import LectureMeta, adapter_for, file_sha256, units_to_documents
from app.kb.registry import KBRegistry
from app.kb.vision import make_image_extractor
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex, build_dense_index
from app.retrieval.encoders import Encoder
from app.retrieval.passages import split_document
from app.retrieval.store import MetadataStore


class KBIndexer:
    def __init__(self, root: Path | str, encoder: Encoder, *, registry: KBRegistry | None = None,
                 target_words: int = 200, max_words: int = 300, image_text=None,
                 image_config: dict | None = None):  # fmt: skip
        self.root = Path(root)
        self.files_dir = self.root / "files"
        self.index_dir = self.root / "index"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder
        self.registry = registry or KBRegistry(self.root / "registry.sqlite")
        self.target_words, self.max_words = target_words, max_words
        self._lock = threading.RLock()
        self.store = MetadataStore(self.index_dir / "metadata.sqlite")
        self.bm25 = BM25Index(self.index_dir / "bm25")
        self._dense: DenseIndex | None = None
        self.jobs: dict[str, dict[str, Any]] = {}
        self.image_text = (image_text if image_text is not None
                           else make_image_extractor(image_config))  # fmt: skip

    # ---- queries use these ---------------------------------------------------------------
    @property
    def dense(self) -> DenseIndex | None:
        if self._dense is None and (self.index_dir / "dense" / "index.faiss").exists():
            self._dense = DenseIndex(self.index_dir / "dense", self.encoder, mmap=False)
        return self._dense

    def has_content(self) -> bool:
        return self.store.counts().get("passages", 0) > 0

    # ---- ingestion -----------------------------------------------------------------------
    def add_file(self, kb_id: str, src: Path, *, file_name: str, uploaded_by: str,
                 background: bool = False, meta: dict | None = None) -> dict[str, Any]:  # fmt: skip
        """Copy the upload under the KB, register it, and index it (optionally in a thread).
        ``meta``: course_id / lecture_id / lecture_title / academic_year / version for decks."""
        lecture = LectureMeta.from_mapping(meta, file_name=file_name)
        sha = file_sha256(src)
        file_id = sha[:16]
        ext = Path(file_name).suffix.lower()
        kind = adapter_for(file_name).kind
        dest_dir = self.files_dir / kb_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{file_id}{ext}"
        if not dest.exists():
            shutil.copyfile(src, dest)
        rec = self.registry.upsert_file(kb_id, file_id=file_id, file_name=file_name, kind=kind,
                                        sha256=sha, size=dest.stat().st_size,
                                        uploaded_by=uploaded_by, path=str(dest),
                                        meta=lecture.__dict__)  # fmt: skip
        job_id = f"{kb_id}:{file_id}"
        self.jobs[job_id] = {"status": "queued", "started_at": time.time(), "file_id": file_id,
                             "kb_id": kb_id}  # fmt: skip
        if background:
            threading.Thread(target=self._index_file, args=(kb_id, file_id, dest, file_name,
                                                              uploaded_by, lecture),
                             daemon=True).start()
        else:
            self._index_file(kb_id, file_id, dest, file_name, uploaded_by, lecture)
            rec = self.registry.get_file(kb_id, file_id) or rec
        return {**rec, "job_id": job_id}

    def _index_file(self, kb_id: str, file_id: str, path: Path, file_name: str,
                    uploaded_by: str, lecture: LectureMeta | None = None) -> None:  # fmt: skip
        job = self.jobs[f"{kb_id}:{file_id}"]
        job["status"] = "indexing"
        self.registry.set_file_status(kb_id, file_id, "indexing")
        try:
            with self._lock:
                adapter = adapter_for(path, image_text=self.image_text)
                units = list(adapter.extract(path))
                docs = units_to_documents(units, kb_id=kb_id, file_id=file_id,
                                          file_name=file_name, kind=adapter.kind,
                                          uploaded_by=uploaded_by, lecture=lecture)  # fmt: skip
                # replace any earlier units of the same file (re-upload / changed content)
                self.store.delete_documents(self.store.doc_ids_where(prefix=f"file:{file_id}:"))
                self._add_documents(docs)
                # deck versioning: earlier versions of the same lecture leave search but stay
                # resolvable for historical citations
                superseded = 0
                if lecture and lecture.lecture_id:
                    for old in self.registry.previous_versions(kb_id, lecture.course_id,
                                                               lecture.lecture_id, file_id):
                        ids = self.store.doc_ids_where(prefix=f"file:{old['file_id']}:")
                        superseded += self.store.mark_superseded(ids, file_id)
                        self.registry.set_file_meta(kb_id, old["file_id"], superseded_by=file_id)
                    if superseded:
                        self._rebuild_bm25()
                self._rebuild_dense()
                needs_ocr = sum(1 for d in docs if d.file and d.file.needs_ocr)
                n_images = sum(d.file.n_images for d in docs if d.file)
                self.registry.set_file_status(kb_id, file_id, "indexed", n_units=len(docs))
                self.registry.set_file_meta(kb_id, file_id, needs_ocr_units=needs_ocr,
                                            n_images=n_images, image_extractor=self.image_text.name,
                                            superseded_units=superseded)  # fmt: skip
                self.registry.bump_version(kb_id)
            job.update(status="done", n_units=len(docs), needs_ocr_units=needs_ocr,
                       finished_at=time.time())  # fmt: skip
        except Exception as e:  # noqa: BLE001  (the job log carries the reason; nothing crashes)
            self.registry.set_file_status(kb_id, file_id, "failed", error=str(e)[:500])
            job.update(status="failed", error=str(e)[:500], finished_at=time.time())

    def _add_documents(self, docs: list[Document]) -> None:
        if not docs:
            return
        self.store.add_documents(docs)
        passages = []
        for d in docs:
            passages.extend(split_document(d, target_words=self.target_words,
                                           max_words=self.max_words))  # fmt: skip
        self.store.add_passages(passages)
        meta = {d.doc_id: (d.year, d.source_type.value, d.course_id) for d in docs}
        self.bm25.add(passages, meta)

    def remove_file(self, kb_id: str, file_id: str) -> int:
        with self._lock:
            ids = self.store.doc_ids_where(prefix=f"file:{file_id}:")
            self.store.delete_documents(ids)
            rec = self.registry.get_file(kb_id, file_id)
            if rec and Path(rec["path"]).exists():
                Path(rec["path"]).unlink()
            self.registry.delete_file(kb_id, file_id)
            self._rebuild_bm25()
            self._rebuild_dense()
            self.registry.bump_version(kb_id)
        return len(ids)

    def remove_kb(self, kb_id: str) -> int:
        with self._lock:
            ids = self.store.doc_ids_where(course_id=kb_id)
            self.store.delete_documents(ids)
            shutil.rmtree(self.files_dir / kb_id, ignore_errors=True)
            self.registry.delete(kb_id)
            self._rebuild_bm25()
            self._rebuild_dense()
        return len(ids)

    # ---- index maintenance ---------------------------------------------------------------
    def _rebuild_bm25(self) -> None:
        shutil.rmtree(self.index_dir / "bm25", ignore_errors=True)
        self.bm25 = BM25Index(self.index_dir / "bm25")
        docs = {d: self.store.get_document(d) for d in self.store.doc_ids_where()}
        meta = {k: (v.year, v.source_type.value, v.course_id) for k, v in docs.items() if v}
        self.bm25.add(list(self.store.iter_passages()), meta)

    def _rebuild_dense(self) -> None:
        self._dense = None
        out = self.index_dir / "dense"
        ids, texts = [], []
        for p in self.store.iter_passages():
            ids.append(p.passage_id)
            texts.append(p.text)
        if not ids:
            shutil.rmtree(out, ignore_errors=True)
            return
        vecs = self.encoder.encode(texts, batch_size=64).astype("float32")
        build_dense_index(iter([vecs]), ids, out, encoder_name=self.encoder.name,
                          dim=self.encoder.dim)  # fmt: skip
        (out / "encoder.json").write_text(json.dumps({"encoder": self.encoder.name,
                                                       "dim": self.encoder.dim}))  # fmt: skip

    def status(self) -> dict[str, Any]:
        return {"counts": self.store.counts(), "jobs": {k: v for k, v in self.jobs.items()
                                                        if v["status"] != "done"}}  # fmt: skip
