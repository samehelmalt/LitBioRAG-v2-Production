"""Shared 60-document fixture corpus: four planted documents plus filler.

Planted cases mirror thesis retrieval failures (PRODUCTION_AUDIT.md §6): a gene-symbol near miss
(MAP3K3 vs MAP3K7), a slash-joined drug name, and a bracketed acronym in a preprint.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from app.core.documents import Document, SourceType

PLANTED = {
    "pmid:900001": (
        "MAP3K3 regulates NF-kappaB activation in endothelial cells",
        "We show that MAP3K3 (MEKK3) is required for NF-kappaB signalling. Knockdown of MAP3K3 "
        "reduced p65 nuclear translocation by 61%.",
    ),
    "pmid:900002": (
        "MAP3K7 (TAK1) in innate immunity",
        "MAP3K7, also known as TAK1, mediates TLR signalling. This study does not examine MAP3K3.",
    ),
    "pmid:900003": (
        "Trifluridine/tipiracil versus fluoropyrimidines in colorectal cancer",
        "Trifluridine/tipiracil (TAS-102) retained activity in 5-FU refractory tumours because "
        "trifluridine is incorporated into DNA rather than inhibiting thymidylate synthase.",
    ),
    "doi:10.1101/2024.05.01.591234": (
        "N6-methyladenosine (m6A) controls mRNA stability",
        "m6A marks on mRNA alter stability, splicing and translation efficiency.",
    ),
}

# (question, gold doc ids, question_type) for runner tests
PLANTED_QUERIES = [
    ("What does MAP3K3 do in NF-kappaB signalling?", ["900001"], "factoid"),
    ("Why is Trifluridine/Tipiracil active in colorectal cancer?", ["900003"], "causal"),
    ("Does MAP3K7 mediate TLR signalling?", ["900002"], "yesno"),
    ("Which mRNA properties does N6-methyladenosine (m6A) affect?",
     ["doi:10.1101/2024.05.01.591234"], "factoid"),
]  # fmt: skip


def fixture_documents() -> Iterator[Document]:
    for doc_id, (title, abstract) in PLANTED.items():
        if doc_id.startswith("pmid:"):
            yield Document(doc_id=doc_id, pmid=doc_id[5:], title=title, abstract=abstract,
                           source_type=SourceType.JOURNAL_ARTICLE, year=2016)  # fmt: skip
        else:
            yield Document(doc_id=doc_id, doi=doc_id[4:], title=title, abstract=abstract,
                           source_type=SourceType.PREPRINT, year=2024,
                           source="biorxiv")  # fmt: skip
    for i in range(56):  # filler on unrelated topics
        yield Document(
            doc_id=f"pmid:{100 + i}", pmid=str(100 + i), year=2015 + i % 10,
            title=f"Study {i} of hip fracture rehabilitation outcomes",
            abstract=f"Cohort {i}: physiotherapy after hip fracture improved walking distance "
            f"by {10 + i} metres at six weeks. Adherence was {50 + i} percent.",
            source_type=SourceType.OBSERVATIONAL,
        )  # fmt: skip


def write_fixture_jsonl(path: Path) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        for d in fixture_documents():
            fh.write(d.model_dump_json() + "\n")
    return path
