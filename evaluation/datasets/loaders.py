"""Loaders for the benchmark query sets.

All loaders return ``QueryRecord`` objects with normalized PMIDs and a stable id, so that
splits, metrics and reports never depend on the upstream file layout.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

THESIS_DIR = Path(__file__).resolve().parent / "thesis"

_PMID_IN_URL = re.compile(r"/pubmed/(\d+)")
_PMID_LABEL = re.compile(r"PMID[:\s]*(\d+)", re.IGNORECASE)

QUESTION_TYPES = ("yesno", "factoid", "list", "summary", "causal", "comparative", "normative")


@dataclass(frozen=True)
class QueryRecord:
    id: str
    question: str
    gold_pmids: tuple[str, ...]
    dataset: str
    question_type: str
    answer: str = ""
    yn_label: str = ""
    year: int | None = None
    source_id: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gold_pmids"] = list(self.gold_pmids)
        return d


def normalize_pmid(value: object) -> str:
    """Return the digits of a PMID given as a number, string or PubMed URL ('' if none)."""
    s = str(value or "").strip()
    m = _PMID_IN_URL.search(s) or _PMID_LABEL.search(s)
    if m:
        return m.group(1)
    return re.sub(r"\D", "", s)


def normalize_question(text: str) -> str:
    """Normalization used for duplicate detection across splits."""
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def query_id(question: str, dataset: str) -> str:
    h = hashlib.sha1(f"{dataset}|{normalize_question(question)}".encode()).hexdigest()
    return f"{dataset.lower()}-{h[:12]}"


def _norm_question_type(raw: str) -> str:
    t = (raw or "").strip().lower().replace("/", "").replace("-", "").replace(" ", "")
    if t in ("yesno", "yn"):
        return "yesno"
    if t in QUESTION_TYPES:
        return t
    return "unknown"


def _yn_label(decision: str) -> str:
    d = (decision or "").strip().lower()
    return d if d in ("yes", "no", "maybe") else ""


def load_thesis_queries(path: Path | None = None) -> list[QueryRecord]:
    """The 24 queries of the thesis master run (``{query, pmids[]}``)."""
    path = path or THESIS_DIR / "benchmark_24_queries.json"
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for r in rows:
        q = r["query"]
        pmids = tuple(p for p in (normalize_pmid(x) for x in r.get("pmids", [])) if p)
        out.append(
            QueryRecord(
                id=query_id(q, "thesis24"),
                question=q,
                gold_pmids=pmids,
                dataset="thesis24",
                question_type="unknown",
            )
        )
    return out


def load_extracted(path: Path, dataset: str) -> list[QueryRecord]:
    """BioASQ / PubMedQA records in the thesis 'extracted queries' schema."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for r in rows:
        q = r.get("question", "")
        if not q:
            continue
        pmids = tuple(p for p in (normalize_pmid(x) for x in r.get("Documents", [])) if p)
        year = r.get("year")
        out.append(
            QueryRecord(
                id=query_id(q, dataset),
                question=q,
                gold_pmids=pmids,
                dataset=dataset,
                question_type=_norm_question_type(r.get("question_type") or r.get("queryType")),
                answer=str(r.get("answer") or ""),
                yn_label=_yn_label(r.get("decision", "")),
                year=int(year) if isinstance(year, int) else None,
                source_id=str(r.get("questionId") or ""),
                extra={k: r[k] for k in ("sourceFormat", "dr_type") if k in r},
            )
        )
    return out


def load_bioasq(path: Path | None = None) -> list[QueryRecord]:
    return load_extracted(path or THESIS_DIR / "bioasq_deduped.json", "BioASQ")


def load_pubmedqa(path: Path | None = None) -> list[QueryRecord]:
    return load_extracted(path or THESIS_DIR / "pubmed_deduped.json", "PubMedQA")


def dedupe(records: Iterable[QueryRecord]) -> list[QueryRecord]:
    """Drop records whose normalized question text repeats; first occurrence wins."""
    seen: set[str] = set()
    out = []
    for r in records:
        key = normalize_question(r.question)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def load_all() -> list[QueryRecord]:
    """BioASQ + PubMedQA, de-duplicated by question text (thesis 24 are a subset)."""
    return dedupe([*load_bioasq(), *load_pubmedqa()])
