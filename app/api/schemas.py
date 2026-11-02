"""Request/response models. The response is the trust surface the UI renders (UI_PROTOTYPE_REVIEW
requirements 1–7): delivery action and deciding rule, per-claim verdicts with evidence spans,
removed claims, conflicts, question-entity grounding, source provenance with tier flags, usage,
versions and the AI disclosure. Nothing here is recomputed client-side."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

DISCLOSURE = (
    "AI-generated summary of retrieved literature. Verified claim by claim against the cited "
    "passages; not medical advice. Check the sources before acting on any statement."
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000)
    course_ids: list[str] | None = Field(default=None, max_length=20)
    max_units: int | None = Field(default=None, ge=1, le=50)
    scope: str | None = Field(default=None, pattern="^(indexed|uploaded|both)$",
                              description="indexed corpus (PubMed, bioRxiv/medRxiv) | uploaded "
                                          "knowledge bases | both (default from api.yaml)")
    kb_ids: list[str] | None = Field(default=None, max_length=20,
                                     description="knowledge bases to search; default: all the "
                                                 "caller can read")
    chat_id: str | None = Field(default=None, max_length=40,
                                description="append to this chat; earlier questions become "
                                            "context for the generator")


class ChatCreate(BaseModel):
    title: str = Field("", max_length=120)


class KeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


class EntityToolIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    question: str = Field("", max_length=1000)


class VerifyToolIn(BaseModel):
    claim: str = Field(..., min_length=3, max_length=2000)
    passage: str = Field(..., min_length=3, max_length=8000)


class ProfileSwitch(BaseModel):
    profile: str = Field(..., min_length=1, max_length=60)


class CorpusActivate(BaseModel):
    corpus_id: str = Field(..., min_length=1, max_length=120)


class CorpusEnabled(BaseModel):
    enabled: bool


class CorpusJobIn(BaseModel):
    type: str = Field(..., pattern="^(fetch|verify|update|fetch_preprints|rebuild)$")
    params: dict[str, Any] = Field(default_factory=dict)


class KBCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field("", max_length=1000)


class KBShare(BaseModel):
    subs: list[str] = Field(default_factory=list, max_length=200)


class KBOut(BaseModel):
    kb_id: str
    name: str
    owner: str
    created_at: float
    description: str | None
    shared_with: list[str]
    version: int
    files: list[dict[str, Any]] = Field(default_factory=list)


class Substitution(BaseModel):
    claim_entity: str
    asked_entity: str
    reason: str | None = None


class EvidenceSpan(BaseModel):
    passage_id: str
    char_start: int
    char_end: int


class ClaimOut(BaseModel):
    id: str
    text: str
    critical: bool
    label: str  # supported | unsupported | contradicted
    support: float
    contradiction: float
    citations: list[str]
    supporting_ids: list[str]
    fabricated_ids: list[str]
    entity_missing: list[str]
    numeric_mismatches: list[str]
    substitution: Substitution | None = None
    off_target: bool = False
    removed: bool = False
    evidence_span: EvidenceSpan | None = None
    contradicting_span: EvidenceSpan | None = None
    notes: list[str] = Field(default_factory=list)


class SourceOut(BaseModel):
    citation_id: str
    passage_id: str
    doc_id: str
    title: str
    year: int | None
    journal: str | None
    source_type: str
    tier: str  # peer_reviewed | preprint | course_material | other
    is_peer_reviewed: bool
    display_source: str
    url: str | None
    text: str
    char_start: int
    char_end: int
    retrieval_score: float
    rank: int
    course_id: str | None = None
    lecture_title: str | None = None
    slide_number: int | None = None
    kb_id: str | None = None
    file_id: str | None = None
    file_name: str | None = None
    locator: str | None = None
    locator_label: str | None = None
    canonical_id: str | None = None
    deck_version: str | None = None
    superseded_by: str | None = None


class ConflictOut(BaseModel):
    claim_id: str
    supporting_ids: list[str]
    contradicting_ids: list[str]
    description: str
    kind: str = "evidence"  # evidence | lecture_vs_literature
    supporting_tiers: list[str] = Field(default_factory=list)
    contradicting_tiers: list[str] = Field(default_factory=list)


class EntityGrounding(BaseModel):
    asked: list[str]
    missing_in_evidence: list[str]
    missing_in_claims: list[str]
    grounding: float | None
    substitution_detected: bool


class ReviewInfo(BaseModel):
    case_id: str
    status: str  # under_review | released | blocked | expired
    message: str = ("This answer was withheld by the safety gate and sent to a human reviewer. "
                    "You will be able to see the reviewer's decision under My cases.")


class QueryResponse(BaseModel):
    request_id: str
    action: str
    internal_state: str
    rule: str
    reason: str
    confidence: float | None
    confidence_calibrated: bool
    sufficiency: float | None
    answer: str
    abstained: bool
    abstain_reason: str | None
    claims: list[ClaimOut]
    removed_claims: list[str]
    conflicts: list[ConflictOut]
    entities: EntityGrounding
    sources: list[SourceOut]
    verification: dict[str, Any]
    usage: dict[str, Any] | None
    evidence_fit: dict[str, Any] | None
    versions: dict[str, Any]
    timings_ms: dict[str, float]
    trace: list[str]
    scope: str = "indexed"
    kb_ids: list[str] = Field(default_factory=list)
    chat_id: str | None = None
    record_id: str | None = None  # immutable answer record (GET /api/v1/records/{id})
    attempts: list[dict[str, Any]] = Field(default_factory=list)  # attempt ledger summaries
    disclosure: str = DISCLOSURE
    review: ReviewInfo | None = None


class SourceDocument(BaseModel):
    doc_id: str
    citation_id: str
    title: str
    abstract: str
    year: int | None
    journal: str | None
    source_type: str
    tier: str
    is_peer_reviewed: bool
    url: str | None
    pmid: str | None
    doi: str | None
    slide: dict[str, Any] | None = None
    file: dict[str, Any] | None = None


class ReviewDecisionIn(BaseModel):
    decision: str = Field(..., pattern="^(release|block)$")
    edited_answer: str | None = Field(default=None, max_length=8000)
    kept_claim_ids: list[str] | None = Field(default=None, max_length=50)
    notes: str = Field("", max_length=4000)
    confirm: bool = False  # required when re-verification flags claims in an edited release


class ReviewCaseOut(BaseModel):
    case_id: str
    request_id: str
    created_at: float
    sub: str
    question: str
    course_ids: list[str]
    action: str
    internal_state: str
    rule: str
    reason: str
    status: str
    reviewer: str | None
    decided_at: float | None
    reviewer_notes: str | None
    released_answer: str | None
    released_claim_ids: list[str] | None
    reverification: dict[str, Any] | None
    response: dict[str, Any] | None = None  # full trust surface; reviewers only
    human_reviewed: dict[str, Any] | None = None


class FeedbackIn(BaseModel):
    request_id: str = Field(..., min_length=4, max_length=64)
    claim_id: str = Field(..., min_length=1, max_length=32)
    label: str = Field(..., pattern="^(correct|incorrect|unsure)$")
    notes: str = Field("", max_length=2000)


class Health(BaseModel):
    status: str
    bundle_id: str
    profile: str
    gate_calibrated_on: str
    preflight: dict[str, Any]
    ledger_spent_today_usd: float | None


class ErrorOut(BaseModel):
    request_id: str
    error: str
