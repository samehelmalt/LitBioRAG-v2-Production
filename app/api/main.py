"""FastAPI service over ``AnswerPipeline``.

Endpoints (``docs/API.md``): ``GET /health``, ``GET /api/v1/models``, ``POST /api/v1/query``,
``GET /api/v1/sources/{doc_id}``. Every request gets a request id, an audit line and a structured
log line; errors never leak stack traces; entitlement is applied before retrieval.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import schemas
from app.api.audit import AuditLog
from app.api.auth import AuthError, Principal, authenticate, check_entitlement
from app.api.ratelimit import RateLimiter
from app.api.settings import ApiSettings
from app.core.budget import Pricing, ResourceConfig, UsageLedger
from app.core.documents import Document, SourceType
from app.evidence.selector import EvidenceUnit
from app.review.service import ReleaseNeedsConfirmation, ReviewConfig, ReviewService
from app.service.pipeline import AnswerPipeline, PipelineOutput

log = logging.getLogger("litbiorag.api")
STATIC = Path(__file__).resolve().parents[1] / "web" / "static"
RUNS = Path(__file__).resolve().parents[2] / "evaluation" / "benchmarks" / "runs"


def _tier(source_type: SourceType, peer_reviewed: bool) -> str:
    if source_type == SourceType.COURSE_SLIDE:
        return "course_material"
    if source_type == SourceType.PREPRINT:
        return "preprint"
    return "peer_reviewed" if peer_reviewed else "other"


def _source(u: EvidenceUnit) -> schemas.SourceOut:
    return schemas.SourceOut(
        citation_id=u.citation_id, passage_id=u.passage_id, doc_id=u.doc_id, title=u.title,
        year=u.year, journal=u.journal, source_type=u.source_type.value,
        tier=_tier(u.source_type, u.is_peer_reviewed), is_peer_reviewed=u.is_peer_reviewed,
        display_source=u.display_source, url=u.url, text=u.text, char_start=u.char_start,
        char_end=u.char_end, retrieval_score=u.score, rank=u.doc_rank, course_id=u.course_id,
        lecture_title=u.lecture_title, slide_number=u.slide_number,
    )  # fmt: skip


def _span(pid: str | None, units: dict[str, EvidenceUnit]) -> schemas.EvidenceSpan | None:
    u = units.get(pid or "")
    return schemas.EvidenceSpan(passage_id=u.passage_id, char_start=u.char_start,
                                char_end=u.char_end) if u else None  # fmt: skip


def to_response(out: PipelineOutput, request_id: str, versions: dict[str, Any],
                calibrated: bool) -> schemas.QueryResponse:  # fmt: skip
    units = {u.passage_id: u for u in out.evidence}
    removed = set(out.decision.removed_claims)
    claims = []
    for v in out.report.verdicts:
        sub = None
        if v.substituted_for:
            ce, _, qe = v.substituted_for.partition("->")
            sub = schemas.Substitution(claim_entity=ce, asked_entity=qe,
                                       reason=v.substitution_reason)  # fmt: skip
        claims.append(schemas.ClaimOut(
            id=v.claim_id, text=v.text, critical=v.critical, label=v.label, support=v.support,
            contradiction=v.contradiction, citations=list(v.cited_ids),
            supporting_ids=list(v.supporting_ids), fabricated_ids=list(v.fabricated_ids),
            entity_missing=list(v.entity_missing), numeric_mismatches=list(v.numeric_mismatches),
            substitution=sub, off_target=v.off_target, removed=v.text in removed,
            evidence_span=_span(v.best_passage_id, units),
            contradicting_span=_span(v.contradicting_passage_id, units), notes=list(v.notes),
        ))  # fmt: skip
    rep = out.report
    return schemas.QueryResponse(
        request_id=request_id, action=out.decision.action.value,
        internal_state=out.decision.internal_state, rule=out.decision.rule,
        reason=out.decision.reason, confidence=out.decision.confidence,
        confidence_calibrated=calibrated, sufficiency=out.decision.sufficiency,
        answer=out.decision.answer_text, abstained=out.generated.abstained,
        abstain_reason=out.generated.abstain_reason, claims=claims,
        removed_claims=list(out.decision.removed_claims),
        conflicts=[schemas.ConflictOut(claim_id=c.claim_id, supporting_ids=list(c.supporting_ids),
                                       contradicting_ids=list(c.contradicting_ids),
                                       description=c.description) for c in rep.conflicts],
        entities=schemas.EntityGrounding(
            asked=list(rep.question_entities),
            missing_in_evidence=list(rep.question_entities_missing_in_evidence),
            missing_in_claims=list(rep.question_entities_missing_in_claims),
            grounding=rep.query_entity_grounding, substitution_detected=rep.substitution_detected,
        ),
        sources=[_source(u) for u in out.evidence],
        verification=rep.model_dump(exclude={"verdicts", "conflicts", "policy_trace"}),
        usage=out.generated.usage, evidence_fit=out.generated.evidence_fit, versions=versions,
        timings_ms=out.timings_ms, trace=list(out.decision.trace),
    )  # fmt: skip


def _doc_out(d: Document) -> schemas.SourceDocument:
    from app.evidence.selector import EvidenceUnit as _EU  # citation id logic lives there

    s = d.slide
    cit = _EU(passage_id=f"{d.doc_id}#0", doc_id=d.doc_id, pmid=d.pmid, doi=d.doi, title=d.title,
              year=d.year, journal=d.journal, source_type=d.source_type,
              is_peer_reviewed=d.is_peer_reviewed, text="", char_start=0, char_end=0, score=0.0,
              doc_rank=0, url=d.url, course_id=s.course_id if s else None,
              lecture_id=s.lecture_id if s else None, lecture_title=s.lecture_title if s else None,
              slide_number=s.slide_number if s else None).citation_id  # fmt: skip
    return schemas.SourceDocument(
        doc_id=d.doc_id, citation_id=cit, title=d.title, abstract=d.abstract, year=d.year,
        journal=d.journal, source_type=d.source_type.value,
        tier=_tier(d.source_type, d.is_peer_reviewed), is_peer_reviewed=d.is_peer_reviewed,
        url=d.url, pmid=d.pmid, doi=d.doi, slide=s.model_dump() if s else None,
    )  # fmt: skip


# ----------------------------------------------------------------------------- app ---------------
def _case_out(case: dict, *, full: bool) -> schemas.ReviewCaseOut:
    hr = ({"reviewer": case["reviewer"], "decided_at": case["decided_at"]}
          if case["status"] == "released" else None)  # fmt: skip
    fields = {k: v for k, v in case.items() if k not in ("evidence", "response")}
    return schemas.ReviewCaseOut(**fields, response=case["response"] if full else None,
                                 human_reviewed=hr)  # fmt: skip


def create_app(settings: ApiSettings | None = None, *, pipeline: AnswerPipeline | None = None,
               ledger: UsageLedger | None = None, review: ReviewService | None = None,
               review_config: ReviewConfig | None = None) -> FastAPI:  # fmt: skip
    st = settings or ApiSettings.load()
    res = ResourceConfig.from_yaml()
    api_cfg = res.api or {}
    if ledger is None:
        lp = api_cfg.get("ledger_path")
        ledger = UsageLedger(
            Path(lp) if lp else None,
            max_usd_per_request=float(api_cfg.get("max_usd_per_request", 0) or 0),
            max_usd_per_day=float(api_cfg.get("max_usd_per_day", 0) or 0),
            pricing=Pricing.load(),
        )
    limiter = RateLimiter(st.limits.rate_per_minute, st.limits.burst)
    audit = AuditLog(st.audit_path, store_question_text=st.audit.store_question_text)
    feedback_path = st.audit_path.parent / "feedback.jsonl" if st.audit_path else None
    feedback_log = AuditLog(feedback_path)
    state: dict[str, Any] = {"pipeline": pipeline, "review": review}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["pipeline"] is None:
            if not st.bundle:
                raise RuntimeError("no bundle configured: set LITBIORAG_BUNDLE or api.yaml bundle")
            state["pipeline"] = AnswerPipeline(
                st.bundle_path, profile=st.profile or None, ledger=ledger, resources=res,
                scope_profile=st.scope_profile,
            )
        if state["review"] is None:
            state["review"] = ReviewService(state["pipeline"], review_config, audit=audit)
        elif state["review"].audit.path is None:
            state["review"].audit = audit  # one audit trail for queries and review decisions
        log.info("startup: %s", state["pipeline"].versions.as_dict())
        yield

    app = FastAPI(title="LitBioRAG v2", version="v1", lifespan=lifespan, docs_url="/api/docs",
                  redoc_url=None)  # fmt: skip
    if st.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=st.cors_origins, allow_methods=["*"],
                           allow_headers=["*"])  # fmt: skip

    def pipe() -> AnswerPipeline:
        p = state["pipeline"]
        if p is None:
            raise HTTPException(503, "service starting")
        return p

    def reviews() -> ReviewService:
        r = state["review"]
        if r is None:
            raise HTTPException(503, "service starting")
        return r

    def require_reviewer(p: Principal) -> None:
        if not (p.is_admin or "reviewer" in p.roles):
            raise HTTPException(403, "reviewer role required")

    # ---- middleware: request id + structured access log + no stack traces --------------------
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001
            log.exception("unhandled error request_id=%s", rid)
            response = JSONResponse(
                status_code=500, content={"request_id": rid, "error": "internal error"}
            )
        response.headers["X-Request-ID"] = rid
        log.info('{"event":"access","request_id":"%s","path":"%s","status":%d,"ms":%.1f}',
                 rid, request.url.path, response.status_code, (time.perf_counter() - t0) * 1000)
        return response

    @app.exception_handler(AuthError)
    async def _auth_error(request: Request, exc: AuthError):
        return JSONResponse(status_code=exc.status, content={
            "request_id": getattr(request.state, "request_id", ""), "error": exc.detail})

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={
            "request_id": getattr(request.state, "request_id", ""), "error": str(exc.detail)},
            headers=getattr(exc, "headers", None))

    # ---- dependencies -------------------------------------------------------------------------
    def principal(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Principal:
        p = authenticate(st.auth, authorization=authorization, api_key=x_api_key)
        key = p.sub if not p.anonymous else f"ip:{request.client.host if request.client else '?'}"
        ok, retry = limiter.allow(key)
        if not ok:
            raise HTTPException(429, "rate limit exceeded",
                                headers={"Retry-After": str(int(retry) + 1)})  # fmt: skip
        return p

    Auth = Depends(principal)

    # ---- endpoints ----------------------------------------------------------------------------
    @app.get("/health", response_model=schemas.Health)
    def health():
        p = pipe()
        return schemas.Health(
            status="ok", bundle_id=p.versions.bundle_id, profile=p.versions.profile,
            gate_calibrated_on=p.versions.gate_calibrated_on or "UNCALIBRATED",
            preflight=p.preflight,
            ledger_spent_today_usd=ledger.spent_today if ledger else None,
        )  # fmt: skip

    @app.get("/api/v1/models")
    def models(_p: Principal = Auth):
        p = pipe()
        return {**p.versions.as_dict(), "verifier_thresholds": p.registry.thresholds("verifier"),
                "gate": {"calibrated_on": p.gate_cfg.calibrated_on or "",
                         "costs": p.gate_cfg.costs}}  # fmt: skip

    @app.post("/api/v1/query", response_model=schemas.QueryResponse,
              responses={401: {"model": schemas.ErrorOut}, 403: {"model": schemas.ErrorOut},
                         429: {"model": schemas.ErrorOut}})  # fmt: skip
    def query(req: schemas.QueryRequest, request: Request, p: Principal = Auth):
        rid = request.state.request_id
        if len(req.question) > st.limits.max_question_chars:
            raise HTTPException(422, f"question longer than {st.limits.max_question_chars} chars")
        courses = check_entitlement(p, req.course_ids)
        pl = pipe()
        out = pl.run(req.question, course_ids=courses, request_id=rid,
                     max_units=min(req.max_units or pl.max_units, st.limits.max_units))  # fmt: skip
        resp = to_response(out, rid, pl.versions.as_dict(), bool(pl.gate_cfg.calibrated_on))
        case = reviews().maybe_enqueue(out, resp.model_dump(mode="json"), sub=p.sub,
                                       course_ids=courses)  # fmt: skip
        if case:
            resp.review = schemas.ReviewInfo(case_id=case["case_id"], status="under_review")
        audit.record({
            "event": "query", "request_id": rid, "sub": p.sub, "via": p.via,
            "courses": courses, "question": req.question, "action": resp.action,
            "rule": resp.rule, "n_claims": len(resp.claims), "n_removed": len(resp.removed_claims),
            "substitution": resp.entities.substitution_detected, "sources": len(resp.sources),
            "review_case": case["case_id"] if case else None,
            "usage": resp.usage, "timings_ms": resp.timings_ms,
            "versions": {k: v for k, v in resp.versions.items() if k != "models"},
        })  # fmt: skip
        return resp

    # ---- human review -----------------------------------------------------------------------
    @app.get("/api/v1/review/queue", response_model=list[schemas.ReviewCaseOut])
    def review_queue(status: str = "pending", limit: int = 50, p: Principal = Auth):
        require_reviewer(p)
        rs = reviews()
        rs.expire()
        if status not in ("pending", "released", "blocked", "expired", "all"):
            raise HTTPException(422, "bad status")
        cases = rs.store.list(status=None if status == "all" else status, limit=min(limit, 200))
        return [_case_out(c, full=False) for c in cases]

    @app.get("/api/v1/review/stats")
    def review_stats(p: Principal = Auth):
        require_reviewer(p)
        return reviews().store.counts()

    @app.get("/api/v1/review/{case_id}", response_model=schemas.ReviewCaseOut)
    def review_case(case_id: str, p: Principal = Auth):
        require_reviewer(p)
        c = reviews().store.get(case_id)
        if c is None:
            raise HTTPException(404, "unknown case")
        return _case_out(c, full=True)

    @app.post("/api/v1/review/{case_id}/decision", response_model=schemas.ReviewCaseOut)
    def review_decide(case_id: str, body: schemas.ReviewDecisionIn, p: Principal = Auth):
        require_reviewer(p)
        try:
            c = reviews().decide(
                case_id, decision=body.decision, reviewer=p.sub, notes=body.notes,
                edited_answer=body.edited_answer, kept_claim_ids=body.kept_claim_ids,
                confirm=body.confirm,
            )
        except ReleaseNeedsConfirmation as e:
            return JSONResponse(status_code=409, content={
                "error": "re-verification flagged claims; resend with confirm=true to release",
                "reverification": e.reverification})
        except LookupError as e:
            raise HTTPException(409 if "pending" in str(e) else 404, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return _case_out(c, full=True)

    @app.get("/api/v1/my/cases", response_model=list[schemas.ReviewCaseOut])
    def my_cases(p: Principal = Auth):
        """The asker's own cases: status and, once released, the human-reviewed answer."""
        if p.anonymous:
            raise HTTPException(401, "sign in to see your cases")
        rs = reviews()
        rs.expire()
        return [_case_out(c, full=False) for c in rs.store.list(status=None, sub=p.sub)]

    @app.post("/api/v1/feedback", status_code=202)
    def feedback(fb: schemas.FeedbackIn, request: Request, p: Principal = Auth):
        """Human labels, claim by claim. Written to the feedback JSONL that the labelling protocol
        in evaluation/labels.py imports; never read at runtime."""
        feedback_log.record({"event": "feedback", "request_id": fb.request_id,
                             "claim_id": fb.claim_id, "label": fb.label, "notes": fb.notes,
                             "rater": p.sub, "via": p.via,
                             "api_request_id": request.state.request_id})  # fmt: skip
        return {"status": "recorded"}

    @app.get("/api/v1/metrics")
    def metrics(_p: Principal = Auth):
        """Latest recorded benchmark summaries, or NOT YET MEASURED. Never computed live."""
        out: dict[str, Any] = {}
        if RUNS.exists():
            for summary in sorted(RUNS.glob("*/**/summary.json")):
                try:
                    out[summary.relative_to(RUNS).as_posix()] = json.loads(summary.read_text())
                except ValueError:
                    continue
        return out or {"status": "NOT YET MEASURED"}

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/trust", include_in_schema=False)
    def trust():
        return FileResponse(STATIC / "trust.html")

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/api/v1/sources/{doc_id:path}", response_model=schemas.SourceDocument)
    def source(doc_id: str, p: Principal = Auth):
        pl = pipe()
        d = pl.store.get_document(doc_id)
        if d is None:
            raise HTTPException(404, "unknown source")
        if d.slide is not None:
            check_entitlement(p, [d.slide.course_id])
        return _doc_out(d)

    return app
