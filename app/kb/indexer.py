"""Index uploaded files into a per-server *uploads bundle* (store + BM25 + dense), incrementally.

Layout (``data/kb/``): ``files/<kb_id>/<file_id><ext>`` raw uploads, ``index/metadata.sqlite``,
``index/bm25``, ``index/dense``. Documents carry ``file`` provenance and ``course_id = kb_id``,
so the existing entitlement filter restricts retrieval to the caller's knowledge bases. BM25 is
appended to in place; the dense index is rebuilt from the store's passages after each change
(small enough for uploads; the main corpus stays a build-host job).
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Any

from app.core.documents import Document
from app.kb.adapters import LectureMeta, adapter_for, file_sha256, units_to_documents
from app.kb.dense_store import IncrementalDense
from app.kb.registry import KBRegistry
from app.kb.vision import make_image_extractor
from app.retrieval.bm25 import BM25Index
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
        self._dense = IncrementalDense(self.index_dir / "dense", encoder)
        self.jobs = self.registry.jobs  # persistent: survives restarts
        if self._dense.ntotal == 0 and self.store.counts().get("passages", 0) > 0:
            self._rebuild_dense()  # encoder changed (or first run on an old layout): re-embed
        self.image_text = (image_text if image_text is not None
                           else make_image_extractor(image_config))  # fmt: skip

    # ---- queries use these ---------------------------------------------------------------
    @property
    def dense(self) -> IncrementalDense:
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
        self.jobs.start(job_id, kb_id=kb_id, file_id=file_id)
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
        job_id = f"{kb_id}:{file_id}"
        self.jobs.update(job_id, status="indexing")
        self.registry.set_file_status(kb_id, file_id, "indexing")
        try:
            with self._lock:
                adapter = adapter_for(path, image_text=self.image_text)
                units = list(adapter.extract(path))
                docs = units_to_documents(units, kb_id=kb_id, file_id=file_id,
                                          file_name=file_name, kind=adapter.kind,
                                          uploaded_by=uploaded_by, lecture=lecture)  # fmt: skip
                # replace any earlier units of the same file (re-upload / changed content)
                self._remove_docs(self.store.doc_ids_where(prefix=f"file:{file_id}:"))
                self._add_documents(docs)
                # deck versioning: earlier versions of the same lecture leave search but stay
                # resolvable for historical citations
                superseded = 0
                if lecture and lecture.lecture_id:
                    for old in self.registry.previous_versions(kb_id, lecture.course_id,
                                                               lecture.lecture_id, file_id):
                        ids = self.store.doc_ids_where(prefix=f"file:{old['file_id']}:")
                        pids = [p.passage_id for d in ids for p in self.store.passages_of(d)]
                        superseded += self.store.mark_superseded(ids, file_id)
                        self.bm25.delete(pids)
                        self._dense.remove(pids)
                        self.registry.set_file_meta(kb_id, old["file_id"], superseded_by=file_id)
                needs_ocr = sum(1 for d in docs if d.file and d.file.needs_ocr)
                n_images = sum(d.file.n_images for d in docs if d.file)
                self.registry.set_file_status(kb_id, file_id, "indexed", n_units=len(docs))
                self.registry.set_file_meta(kb_id, file_id, needs_ocr_units=needs_ocr,
                                            n_images=n_images, image_extractor=self.image_text.name,
                                            superseded_units=superseded)  # fmt: skip
                self.registry.bump_version(kb_id)
            self.jobs.update(job_id, status="done", n_units=len(docs), needs_ocr_units=needs_ocr,
                             finished_at=time.time())  # fmt: skip
        except Exception as e:  # noqa: BLE001  (the job log carries the reason; nothing crashes)
            self.registry.set_file_status(kb_id, file_id, "failed", error=str(e)[:500])
            self.jobs.update(job_id, status="failed", error=str(e)[:500],
                             finished_at=time.time())  # fmt: skip

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
        self._dense.add([p.passage_id for p in passages], [p.text for p in passages])

    def _remove_docs(self, doc_ids: list[str]) -> list[str]:
        """Remove documents and their passages from the store and both indexes."""
        pids = [p.passage_id for d in doc_ids for p in self.store.passages_of(d)]
        self.store.delete_documents(doc_ids)
        self.bm25.delete(pids)
        self._dense.remove(pids)
        return pids

    def remove_file(self, kb_id: str, file_id: str) -> int:
        with self._lock:
            ids = self.store.doc_ids_where(prefix=f"file:{file_id}:")
            self._remove_docs(ids)
            rec = self.registry.get_file(kb_id, file_id)
            if rec and Path(rec["path"]).exists():
                Path(rec["path"]).unlink()
            self.registry.delete_file(kb_id, file_id)
            self.registry.bump_version(kb_id)
        return len(ids)

    def remove_kb(self, kb_id: str) -> int:
        with self._lock:
            ids = self.store.doc_ids_where(course_id=kb_id)
            self._remove_docs(ids)
            shutil.rmtree(self.files_dir / kb_id, ignore_errors=True)
            self.registry.delete(kb_id)
        return len(ids)

    # ---- index maintenance ---------------------------------------------------------------
    def _rebuild_dense(self) -> None:
        """Full re-embed from the store: only when the encoder changed."""
        with self._lock:
            shutil.rmtree(self.index_dir / "dense", ignore_errors=True)
            self._dense = IncrementalDense(self.index_dir / "dense", self.encoder)
            ps = list(self.store.iter_passages())
            if ps:
                self._dense.add([p.passage_id for p in ps], [p.text for p in ps])

    def status(self) -> dict[str, Any]:
        return {"counts": self.store.counts(), "dense_vectors": self._dense.ntotal,
                "jobs": {k: v for k, v in self.jobs.items() if v["status"] != "done"}}  # fmt: skip
