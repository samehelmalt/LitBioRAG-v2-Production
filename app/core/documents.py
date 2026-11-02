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
    COURSE_SLIDE = "course_slide"  # teaching material (lecture slide); not peer-reviewed literature
    UPLOADED_DOCUMENT = "uploaded_document"  # user-uploaded file (PDF/DOCX/XLSX/PPTX/TXT/MD)
    OTHER = "other"


EVIDENCE_RANK: dict[SourceType, int] = {
    SourceType.SYSTEMATIC_REVIEW: 0,
    SourceType.RCT: 1,
    SourceType.CLINICAL_TRIAL: 2,
    SourceType.OBSERVATIONAL: 3,
    SourceType.JOURNAL_ARTICLE: 4,
    SourceType.REVIEW: 4,
    SourceType.CASE_REPORT: 5,
    SourceType.COURSE_SLIDE: 5,  # authoritative *for the course*, not as literature
    SourceType.UPLOADED_DOCUMENT: 5,  # trusted by its owner; provenance unknown to the system
    SourceType.PREPRINT: 6,
    SourceType.OTHER: 7,
}

PEER_REVIEWED = frozenset(
    t
    for t in SourceType
    if t not in (SourceType.PREPRINT, SourceType.COURSE_SLIDE, SourceType.UPLOADED_DOCUMENT,
                 SourceType.OTHER)
)


class FileMeta(BaseModel):
    """Provenance of one unit of an uploaded file: what 'file name, page N' resolves to."""

    kb_id: str
    file_id: str  # sha256[:16] of the file content
    file_name: str
    kind: str  # pdf | docx | xlsx | pptx | text
    locator_kind: str  # page | section | sheet | slide | text
    locator: str  # "p3" | "s7" | "Sheet1!A1:D40" | "sec2" | "t1"
    label: str = ""  # human label: "page 3", "Sheet1 rows 1–40", "slide 7", "section 'Dosing'"
    page_number: int | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    heading: str = ""
    version: str = ""
    content_hash: str = ""
    uploaded_by: str = ""
    # lecture provenance for uploaded decks (commercial contract: every citation resolves to the
    # exact slide of a known lecture version)
    course_id: str = ""
    lecture_id: str = ""
    lecture_title: str = ""
    academic_year: str = ""
    deck_version: str = ""
    slide_number: int | None = None
    section: str = ""
    n_images: int = 0
    images_with_text: int = 0
    needs_ocr: bool = False  # pictures present but no text could be extracted from them
    superseded_by: str = ""  # file_id of a newer version of the same lecture

    @property
    def canonical_id(self) -> str:
        """course_id/lecture_id/slide/version/content_hash (for files without lecture metadata:
        kb_id/file_id/locator/version/hash)."""
        if self.lecture_id:
            return (f"{self.course_id or '-'}/{self.lecture_id}/s{self.slide_number or 0}/"
                    f"v{self.deck_version or self.version or '1'}/{self.content_hash}")
        return (f"{self.kb_id}/{self.file_id}/{self.locator}/v{self.version or '1'}/"
                f"{self.content_hash}")


class SlideMeta(BaseModel):
    """Provenance of a lecture slide; what a 'Lecture title, Slide N' citation resolves to."""

    course_id: str
    lecture_id: str
    lecture_title: str
    slide_number: int
    file_name: str = ""
    slide_title: str = ""
    section: str = ""
    academic_year: str = ""
    version: str = ""
    content_hash: str = ""
    n_images: int = 0
    images_with_text: int = 0
    needs_ocr: bool = False

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
    source: str = Field(default="pubmed", description="pubmed | biorxiv | medrxiv | pptx")
    version: str | None = Field(default=None, description="preprint version, e.g. '2'")
    slide: SlideMeta | None = None
    file: FileMeta | None = None

    @property
    def course_id(self) -> str | None:
        """Entitlement key: the course for slides, the knowledge base for uploaded files."""
        if self.slide:
            return self.slide.course_id
        if self.file:
            return self.file.kb_id
        return None

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

    @staticmethod
    def make_slide_id(lecture_id: str, slide_number: int) -> str:
        return f"slide:{lecture_id}:{int(slide_number)}"

    @staticmethod
    def make_file_id(file_id: str, locator: str) -> str:
        return f"file:{file_id}:{locator}"


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
