"""Typed corpus records shared by ingestion, indexing, retrieval and the API.

A ``Document`` is one bibliographic record (a PubMed abstract or a preprint). A ``Passage`` is
a retrieval unit cut from a document with character offsets, so that any evidence span shown to
a user or a verifier can be traced back to its exact location in the source text.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    """Publication type used by the evidence hierarchy. Lower rank = stronger source class."""

    SYSTEMATIC_REVIEW = "systematic_review"  # incl. meta-analysis
    RCT = "randomized_controlled_trial"
    CLINICAL_TRIAL = "clinical_trial"
    OBSERVATIONAL = "observational_study"
    CASE_REPORT = "case_report"
    REVIEW = "review"
    JOURNAL_ARTICLE = "journal_article"  # peer-reviewed, type not further classified
    PREPRINT = "preprint"  # not peer-reviewed
    OTHER = "other"


EVIDENCE_RANK: dict[SourceType, int] = {
    SourceType.SYSTEMATIC_REVIEW: 0,
    SourceType.RCT: 1,
    SourceType.CLINICAL_TRIAL: 2,
    SourceType.OBSERVATIONAL: 3,
    SourceType.JOURNAL_ARTICLE: 4,
    SourceType.REVIEW: 4,
    SourceType.CASE_REPORT: 5,
    SourceType.PREPRINT: 6,
    SourceType.OTHER: 7,
}

PEER_REVIEWED = frozenset(t for t in SourceType if t not in (SourceType.PREPRINT, SourceType.OTHER))

# PubMed <PublicationType> strings -> SourceType (first match in this order wins).
_PUBTYPE_MAP: tuple[tuple[str, SourceType], ...] = (
    ("Systematic Review", SourceType.SYSTEMATIC_REVIEW),
    ("Meta-Analysis", SourceType.SYSTEMATIC_REVIEW),
    ("Randomized Controlled Trial", SourceType.RCT),
    ("Clinical Trial", SourceType.CLINICAL_TRIAL),
    ("Observational Study", SourceType.OBSERVATIONAL),
    ("Cohort Studies", SourceType.OBSERVATIONAL),
    ("Case-Control Studies", SourceType.OBSERVATIONAL),
    ("Cross-Sectional Studies", SourceType.OBSERVATIONAL),
    ("Case Reports", SourceType.CASE_REPORT),
    ("Review", SourceType.REVIEW),
    ("Journal Article", SourceType.JOURNAL_ARTICLE),
)


def source_type_from_pubtypes(pub_types: list[str]) -> SourceType:
    """Map PubMed publication types to a SourceType; 'Journal Article' alone -> JOURNAL_ARTICLE."""
    joined = {p.strip() for p in pub_types}
    for key, st in _PUBTYPE_MAP:
        if any(key in p for p in joined):
            return st
    return SourceType.OTHER


class Document(BaseModel):
    doc_id: str = Field(description="'pmid:<n>' for PubMed, 'doi:<doi>' for preprints")
    title: str
    abstract: str
    source_type: SourceType
    pmid: str | None = None
    doi: str | None = None
    year: int | None = None
    journal: str | None = None
    pub_types: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    url: str | None = None
    source: str = Field(default="pubmed", description="ingestion source: pubmed|biorxiv|medrxiv")
    version: str | None = Field(default=None, description="preprint version, e.g. '2'")

    @field_validator("pmid")
    @classmethod
    def _digits(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = "".join(ch for ch in str(v) if ch.isdigit())
        return v or None

    @field_validator("doi")
    @classmethod
    def _lower_doi(cls, v: str | None) -> str | None:
        return v.strip().lower() if v else None

    @property
    def is_peer_reviewed(self) -> bool:
        return self.source_type in PEER_REVIEWED

    @property
    def evidence_rank(self) -> int:
        return EVIDENCE_RANK[self.source_type]

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.abstract}".strip()

    @staticmethod
    def make_id(pmid: str | None, doi: str | None) -> str:
        if pmid:
            return f"pmid:{''.join(ch for ch in pmid if ch.isdigit())}"
        if doi:
            return f"doi:{doi.strip().lower()}"
        raise ValueError("a document needs a PMID or a DOI")


class Passage(BaseModel):
    passage_id: str = Field(description="'<doc_id>#<ord>'")
    doc_id: str
    ord: int
    text: str
    char_start: int = Field(description="offset into Document.text")
    char_end: int
    n_words: int

    @staticmethod
    def make_id(doc_id: str, ord: int) -> str:
        return f"{doc_id}#{ord}"
