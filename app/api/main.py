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
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.api import schemas
from app.api.audit import AuditLog
from app.api.auth import AuthError, Principal, authenticate, check_entitlement
from app.api.chats import ChatStore
from app.api.keys import KeyStore
from app.api.ratelimit import RateLimiter
from app.api.settings import ApiSettings
from app.api.trials import TrialLedger, TrialLimitExceeded
from app.core.admission import Overloaded
from app.core.budget import Pricing, ResourceConfig, UsageLedger
from app.core.documents import Document, SourceType
from app.corpus.manager import CorpusConfig, CorpusManager
from app.evidence.selector import EvidenceUnit
from app.review.service import ReleaseNeedsConfirmation, ReviewConfig, ReviewService
from app.service.pipeline import AnswerPipeline, PipelineOutput
from app.service.records import RecordStore, build_record

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
        lecture_title=u.lecture_title, slide_number=u.slide_number, kb_id=u.kb_id,
        file_id=u.file_id, file_name=u.file_name, locator=u.locator,
        locator_label=u.locator_label, canonical_id=u.canonical_id,
        deck_version=u.deck_version, superseded_by=u.superseded_by,
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
                                       description=c.description, kind=c.kind,
                                       supporting_tiers=list(c.supporting_tiers),
                                       contradicting_tiers=list(c.contradicting_tiers))
                   for c in rep.conflicts],
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
              slide_number=s.slide_number if s else None,
              kb_id=d.file.kb_id if d.file else None, file_id=d.file.file_id if d.file else None,
              file_name=d.file.file_name if d.file else None,
              locator=d.file.locator if d.file else None,
              locator_label=d.file.label if d.file else None).citation_id  # fmt: skip
    return schemas.SourceDocument(
        doc_id=d.doc_id, citation_id=cit, title=d.title, abstract=d.abstract, year=d.year,
        journal=d.journal, source_type=d.source_type.value,
        tier=_tier(d.source_type, d.is_peer_reviewed), is_peer_reviewed=d.is_peer_reviewed,
        url=d.url, pmid=d.pmid, doi=d.doi, slide=s.model_dump() if s else None,
        file=d.file.model_dump() if d.file else None,
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
               review_config: ReviewConfig | None = None,
               corpus_config: CorpusConfig | None = None) -> FastAPI:  # fmt: skip
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
    data_dir = st.audit_path.parent if st.audit_path else None
    chats = ChatStore(data_dir / "chats.sqlite" if data_dir else None)
    keys = KeyStore(data_dir / "api_keys.sqlite" if data_dir else None)
    records = RecordStore(data_dir / "answers.sqlite" if data_dir else None)
    trials: TrialLedger | None = None
    if st.trial.questions > 0:
        tp = Path(st.trial.db_path)
        trials = TrialLedger(tp if tp.is_absolute() else Path(__file__).resolve().parents[2] / tp,
                             st.trial.questions)  # fmt: skip
    state: dict[str, Any] = {"pipeline": pipeline, "review": review, "corpus": None}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["pipeline"] is None:
            if not st.bundle:
                raise RuntimeError("no bundle configured: set LITBIORAG_BUNDLE or api.yaml bundle")
            state["pipeline"] = AnswerPipeline(
                st.bundle_path, profile=st.profile or None, ledger=ledger, resources=res,
                scope_profile=st.scope_profile, kb_root=st.kb_root,
            )
        if state["review"] is None:
            state["review"] = ReviewService(state["pipeline"], review_config, audit=audit)
        elif state["review"].audit.path is None:
            state["review"].audit = audit  # one audit trail for queries and review decisions
        pl = state["pipeline"]
        enc_name = pl.pool.acquire("dense_encoder").name
        state["corpus"] = CorpusManager(corpus_config, activate=pl.swap_bundle,
                                        active_bundle=pl.bundle, encoder_name=enc_name)  # fmt: skip
        log.info("startup: %s", state["pipeline"].versions.as_dict())
        yield

    app = FastAPI(title="AEOS Medical", version="v1", lifespan=lifespan, docs_url="/api/docs",
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

    def corpus() -> CorpusManager:
        c = state["corpus"]
        if c is None:
            raise HTTPException(503, "service starting")
        return c

    def require_admin(p: Principal) -> None:
        if not p.is_admin:
            raise HTTPException(403, "admin role required")

    def require_reviewer(p: Principal) -> None:
        if not (p.is_admin or "reviewer" in p.roles):
            raise HTTPException(403, "reviewer role required")

    def kb_indexer():
        pl = pipe()
        if pl.kb is None:
            raise HTTPException(404, "uploaded knowledge bases are disabled on this server")
        return pl.kb

    def readable_kbs(p: Principal, requested: list[str] | None) -> list[str]:
        """Entitlement for uploads: owner, shared-with, or admin. Unreadable requested → 403."""
        reg = kb_indexer().registry
        mine = [k["kb_id"] for k in reg.list_for(p.sub, admin=p.is_admin)]
        if not requested:
            return mine
        bad = [k for k in requested if k not in mine]
        if bad:
            raise HTTPException(403, f"not entitled to knowledge base(s): {', '.join(bad)}")
        return list(requested)

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
        p = authenticate(st.auth, authorization=authorization, api_key=x_api_key,
                         key_lookup=keys.lookup)  # fmt: skip
        key = p.sub if not p.anonymous else f"ip:{request.client.host if request.client else '?'}"
        ok, retry = limiter.allow(key)
        if not ok:
            raise HTTPException(429, "rate limit exceeded",
                                headers={"Retry-After": str(int(retry) + 1)})  # fmt: skip
        return p

    Auth = Depends(principal)
    Upload = File(...)
    BackgroundFlag = Form(False)

    # ---- endpoints ----------------------------------------------------------------------------
    @app.get("/health", response_model=schemas.Health)
    def health():
        p = pipe()
        return schemas.Health(
            status="ok", bundle_id=p.versions.bundle_id, profile=p.versions.profile,
            gate_calibrated_on=p.versions.gate_calibrated_on or "UNCALIBRATED",
            preflight=p.preflight,
            ledger_spent_today_usd=ledger.spent_today if ledger else None,
            concurrency=p.pool.admission.snapshot(),
        )  # fmt: skip

    @app.get("/api/v1/trial")
    def trial_status(p: Principal = Auth):
        if trials is None:
            return {"enabled": False, "limit": None, "remaining": None}
        if p.anonymous:
            raise HTTPException(401, "sign in to see your trial allowance")
        return {"enabled": True, "limit": trials.limit, "remaining": trials.remaining(p.sub),
                "used": trials.used(p.sub), "exempt": p.is_admin}  # fmt: skip

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
        return run_query(req, request.state.request_id, p)

    def run_query(req: schemas.QueryRequest, rid: str, p: Principal,
                  progress=None) -> schemas.QueryResponse:  # fmt: skip
        if len(req.question) > st.limits.max_question_chars:
            raise HTTPException(422, f"question longer than {st.limits.max_question_chars} chars")
        courses = check_entitlement(p, req.course_ids)
        pl = pipe()
        history: list[str] = []
        chat = None
        if req.chat_id:
            chat = chats.owned(req.chat_id, p.sub) if not p.anonymous else None
            if chat is None:
                raise HTTPException(404, "unknown chat")
            history = chats.history(req.chat_id)
        scope = req.scope or ("uploaded" if st.scope_profile == "education" and pl.kb is not None
                              else st.kb.default_scope)
        kb_ids: list[str] = []
        if not corpus().indexed_enabled:
            if scope == "indexed":
                raise HTTPException(422, "the indexed corpus is disabled on this server; "
                                         "use scope 'uploaded'")  # fmt: skip
            scope = "uploaded"
        if scope != "indexed":
            if pl.kb is None:
                if scope == "uploaded":
                    raise HTTPException(422, "uploaded knowledge bases are disabled on this server")
                scope = "indexed"
            else:
                kb_ids = readable_kbs(p, req.kb_ids)
                if scope == "uploaded" and not kb_ids:
                    raise HTTPException(422, "no knowledge base selected or readable; upload "
                                             "files first or choose scope 'indexed'")  # fmt: skip
        # trial allowance: reserved after every validation, refunded when no answer is produced
        charged = False
        if trials is not None and not p.is_admin:
            if p.anonymous:
                raise HTTPException(401, "sign in to use the trial")
            try:
                trials.reserve(p.sub, rid)
                charged = True
            except TrialLimitExceeded as e:
                raise HTTPException(429, str(e)) from e
        try:
            out = pl.run(req.question, course_ids=courses, request_id=rid,
                         max_units=min(req.max_units or pl.max_units, st.limits.max_units),
                         scope=scope, kb_ids=kb_ids, history=history,
                         progress=progress)  # fmt: skip
        except Overloaded as e:
            if charged:
                trials.refund(p.sub, rid)
            audit.record({"event": "overloaded", "request_id": rid, "sub": p.sub,
                          "role": e.role, "reason": e.reason})  # fmt: skip
            raise HTTPException(
                503, f"the service is busy ({e.role}: {e.reason}); try again shortly",
                headers={"Retry-After": str(int(e.retry_after_s))},
            ) from e
        except Exception:
            if charged:
                trials.refund(p.sub, rid)
            raise
        resp = to_response(out, rid, pl.versions.as_dict(), bool(pl.gate_cfg.calibrated_on))
        resp.scope, resp.kb_ids = scope, kb_ids
        resp.attempts = [a.summary() for a in out.attempts]
        if out.resolution is not None and out.resolution.interpreted:
            resp.interpreted_as = out.resolution.question
            resp.interpretation_note = out.resolution.note
        case = reviews().maybe_enqueue(out, resp.model_dump(mode="json"), sub=p.sub,
                                       course_ids=courses)  # fmt: skip
        if case:
            resp.review = schemas.ReviewInfo(case_id=case["case_id"], status="under_review")
        if chat is not None:
            resp.chat_id = req.chat_id
        # the immutable record is the single source for chat, audit and review views
        rec = build_record(out, resp.model_dump(mode="json"), sub=p.sub, chat_id=req.chat_id,
                           scope=scope, kb_ids=kb_ids, versions=pl.versions.as_dict(),
                           operating_point=pl.gate_cfg.operating_point)  # fmt: skip
        records.save(rec)
        resp.record_id = rid
        if chat is not None:
            chats.append(req.chat_id, request_id=rid, question=req.question, action=resp.action,
                         answer=resp.answer, response=resp.model_dump(mode="json"))  # fmt: skip
        audit.record({
            "event": "query", "request_id": rid, "sub": p.sub, "via": p.via,
            "courses": courses, "question": req.question, "action": resp.action,
            "rule": resp.rule, "n_claims": len(resp.claims), "n_removed": len(resp.removed_claims),
            "substitution": resp.entities.substitution_detected, "sources": len(resp.sources),
            "review_case": case["case_id"] if case else None,
            "scope": scope, "kb_ids": kb_ids,
            "usage": resp.usage, "timings_ms": resp.timings_ms,
            "versions": {k: v for k, v in resp.versions.items() if k != "models"},
        })  # fmt: skip
        return resp

    # ---- knowledge bases (uploaded files) ---------------------------------------------------
    def _kb_out(kb: dict, files: list[dict]) -> schemas.KBOut:
        return schemas.KBOut(**kb, files=[{k: v for k, v in f.items() if k != "path"}
                                          for f in files])  # fmt: skip

    @app.get("/api/v1/kb", response_model=list[schemas.KBOut])
    def kb_list(p: Principal = Auth):
        ix = kb_indexer()
        return [_kb_out(k, ix.registry.files(k["kb_id"]))
                for k in ix.registry.list_for(p.sub, admin=p.is_admin)]  # fmt: skip

    @app.post("/api/v1/kb", response_model=schemas.KBOut, status_code=201)
    def kb_create(body: schemas.KBCreate, p: Principal = Auth):
        if p.anonymous:
            raise HTTPException(401, "sign in to create a knowledge base")
        ix = kb_indexer()
        kb = ix.registry.create(name=body.name, owner=p.sub, description=body.description)
        audit.record({"event": "kb_create", "kb_id": kb["kb_id"], "sub": p.sub})
        return _kb_out(kb, [])

    @app.get("/api/v1/kb/{kb_id}", response_model=schemas.KBOut)
    def kb_get(kb_id: str, p: Principal = Auth):
        ix = kb_indexer()
        if not ix.registry.can_read(kb_id, p.sub, admin=p.is_admin):
            raise HTTPException(403, "not entitled to this knowledge base")
        return _kb_out(ix.registry.get(kb_id), ix.registry.files(kb_id))  # type: ignore[arg-type]

    @app.post("/api/v1/kb/{kb_id}/share", response_model=schemas.KBOut)
    def kb_share(kb_id: str, body: schemas.KBShare, p: Principal = Auth):
        ix = kb_indexer()
        if not ix.registry.can_write(kb_id, p.sub, admin=p.is_admin):
            raise HTTPException(403, "only the owner can share a knowledge base")
        return _kb_out(ix.registry.share(kb_id, body.subs), ix.registry.files(kb_id))

    @app.post("/api/v1/kb/{kb_id}/upload", status_code=202)
    def kb_upload(kb_id: str, file: UploadFile = Upload, background: bool = BackgroundFlag,
                  course_id: Annotated[str, Form()] = "", lecture_id: Annotated[str, Form()] = "",
                  lecture_title: Annotated[str, Form()] = "",
                  academic_year: Annotated[str, Form()] = "",
                  version: Annotated[str, Form()] = "", p: Principal = Auth):  # fmt: skip
        ix = kb_indexer()
        if not ix.registry.can_write(kb_id, p.sub, admin=p.is_admin):
            raise HTTPException(403, "only the owner can upload to this knowledge base")
        if len(ix.registry.files(kb_id)) >= st.limits.max_files_per_kb:
            raise HTTPException(422, "knowledge base file limit reached")
        from app.kb.adapters import SUPPORTED_EXTENSIONS

        name = Path(file.filename or "upload").name
        if not name.lower().endswith(SUPPORTED_EXTENSIONS):
            raise HTTPException(415, f"unsupported file type; allowed: "
                                     f"{', '.join(SUPPORTED_EXTENSIONS)}")  # fmt: skip
        import tempfile

        limit = st.limits.max_upload_mb * 1024 * 1024
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix) as tmp:
            size = 0
            while chunk := file.file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    tmp.close()
                    Path(tmp.name).unlink(missing_ok=True)
                    raise HTTPException(413, f"file larger than {st.limits.max_upload_mb} MB")
                tmp.write(chunk)
        try:
            meta = {k: v for k, v in (("course_id", course_id), ("lecture_id", lecture_id),
                                      ("lecture_title", lecture_title),
                                      ("academic_year", academic_year), ("version", version))
                    if v}
            rec = ix.add_file(kb_id, Path(tmp.name), file_name=name, uploaded_by=p.sub,
                              background=background, meta=meta)  # fmt: skip
        finally:
            Path(tmp.name).unlink(missing_ok=True)
        audit.record({"event": "kb_upload", "kb_id": kb_id, "file_id": rec["file_id"],
                      "file_name": name, "bytes": size, "sub": p.sub,
                      "status": rec["status"]})  # fmt: skip
        return {k: v for k, v in rec.items() if k != "path"}

    @app.get("/api/v1/kb/{kb_id}/status")
    def kb_status(kb_id: str, p: Principal = Auth):
        ix = kb_indexer()
        if not ix.registry.can_read(kb_id, p.sub, admin=p.is_admin):
            raise HTTPException(403, "not entitled to this knowledge base")
        files = [{k: v for k, v in f.items() if k != "path"} for f in ix.registry.files(kb_id)]
        jobs = {k: v for k, v in ix.jobs.items() if v.get("kb_id") == kb_id}
        return {"kb_id": kb_id, "version": ix.registry.get(kb_id)["version"],  # type: ignore
                "files": files, "jobs": jobs}

    @app.get("/api/v1/slides/{file_id}/{n}")
    def slide(file_id: str, n: int, p: Principal = Auth):
        """Resolve a slide citation to the exact slide: text, notes, image texts, neighbours,
        canonical id, version and whether a newer version supersedes it."""
        ix = kb_indexer()
        d = ix.store.get_document(f"file:{file_id}:s{n}")
        if d is None or d.file is None:
            raise HTTPException(404, "unknown slide")
        readable_kbs(p, [d.file.kb_id])
        f = d.file
        neighbours = {}
        for k in (n - 1, n + 1):
            nd = ix.store.get_document(f"file:{file_id}:s{k}")
            if nd is not None:
                neighbours[k] = {"title": nd.title, "canonical_id": nd.file.canonical_id}
        return {
            "canonical_id": f.canonical_id, "citation_id": f"SLIDE:{f.lecture_id}:{n}"
            if f.lecture_id else f"FILE:{file_id}:s{n}", "course_id": f.course_id,
            "lecture_id": f.lecture_id, "lecture_title": f.lecture_title, "slide_number": n,
            "deck_version": f.deck_version, "academic_year": f.academic_year,
            "file_name": f.file_name, "title": d.title, "text": d.abstract, "section": f.section,
            "n_images": f.n_images, "images_with_text": f.images_with_text,
            "needs_ocr": f.needs_ocr, "superseded_by": f.superseded_by or None,
            "neighbours": neighbours, "content_hash": f.content_hash,
        }  # fmt: skip

    @app.delete("/api/v1/kb/{kb_id}/files/{file_id}")
    def kb_delete_file(kb_id: str, file_id: str, p: Principal = Auth):
        ix = kb_indexer()
        if not ix.registry.can_write(kb_id, p.sub, admin=p.is_admin):
            raise HTTPException(403, "only the owner can delete files")
        if ix.registry.get_file(kb_id, file_id) is None:
            raise HTTPException(404, "unknown file")
        n = ix.remove_file(kb_id, file_id)
        audit.record({"event": "kb_delete_file", "kb_id": kb_id, "file_id": file_id, "sub": p.sub})
        return {"removed_units": n}

    @app.delete("/api/v1/kb/{kb_id}")
    def kb_delete(kb_id: str, p: Principal = Auth):
        ix = kb_indexer()
        if not ix.registry.can_write(kb_id, p.sub, admin=p.is_admin):
            raise HTTPException(403, "only the owner can delete a knowledge base")
        n = ix.remove_kb(kb_id)
        audit.record({"event": "kb_delete", "kb_id": kb_id, "sub": p.sub})
        return {"removed_units": n}

    # ---- corpus indexes (admin) ------------------------------------------------------------
    @app.get("/api/v1/corpus")
    def corpus_status(p: Principal = Auth):
        require_admin(p)
        return corpus().status()

    @app.post("/api/v1/corpus/activate")
    def corpus_activate(body: schemas.CorpusActivate, p: Principal = Auth):
        require_admin(p)
        try:
            info = corpus().activate(body.corpus_id)
        except LookupError as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        audit.record({"event": "corpus_activate", "corpus_id": body.corpus_id, "sub": p.sub})
        return info

    @app.post("/api/v1/corpus/enabled")
    def corpus_enabled(body: schemas.CorpusEnabled, p: Principal = Auth):
        require_admin(p)
        corpus().indexed_enabled = body.enabled
        audit.record({"event": "corpus_enabled", "enabled": body.enabled, "sub": p.sub})
        return {"indexed_enabled": body.enabled}

    @app.post("/api/v1/corpus/jobs", status_code=202)
    def corpus_job(body: schemas.CorpusJobIn, p: Principal = Auth):
        require_admin(p)
        try:
            job = corpus().start_job(body.type, body.params, by=p.sub)
        except PermissionError as e:
            raise HTTPException(409, str(e)) from e
        except (ValueError, KeyError) as e:
            raise HTTPException(422, f"bad job: {e}") from e
        audit.record({"event": "corpus_job", "job_id": job["job_id"], "type": body.type,
                      "params": body.params, "sub": p.sub})  # fmt: skip
        return {k: v for k, v in job.items() if k != "log"}

    @app.get("/api/v1/corpus/jobs/{job_id}")
    def corpus_job_status(job_id: str, p: Principal = Auth):
        require_admin(p)
        j = corpus().job(job_id)
        if j is None:
            raise HTTPException(404, "unknown job")
        return {k: v for k, v in j.items() if k != "log"}

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
        return FileResponse(STATIC / "index.html")  # single-page app; route #/trust

    @app.get("/embed", include_in_schema=False)
    def embed():
        return FileResponse(STATIC / "embed.html")

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/api/v1/query/stream")
    def query_stream(question: str, request: Request, scope: str | None = None,
                     chat_id: str | None = None, course_ids: str | None = None,
                     kb_ids: str | None = None, p: Principal = Auth):  # fmt: skip
        """Server-sent events: progress stages, then the same QueryResponse as POST /query."""
        import queue
        import threading

        req = schemas.QueryRequest(
            question=question, scope=scope, chat_id=chat_id,
            course_ids=[c for c in (course_ids or "").split(",") if c] or None,
            kb_ids=[k for k in (kb_ids or "").split(",") if k] or None,
        )
        rid = request.state.request_id
        q: queue.Queue = queue.Queue()

        def progress(stage: str, info: dict) -> None:
            q.put(("stage", {"stage": stage, **info}))

        def work() -> None:
            try:
                resp = run_query(req, rid, p, progress=progress)
                q.put(("result", resp.model_dump(mode="json")))
            except HTTPException as e:
                q.put(("error", {"request_id": rid, "error": str(e.detail),
                                 "status": e.status_code}))  # fmt: skip
            except AuthError as e:
                q.put(("error", {"request_id": rid, "error": e.detail, "status": e.status}))
            except Exception:  # noqa: BLE001
                log.exception("stream error request_id=%s", rid)
                q.put(("error", {"request_id": rid, "error": "internal error", "status": 500}))
            q.put(None)

        threading.Thread(target=work, daemon=True).start()

        def events():
            while True:
                item = q.get()
                if item is None:
                    break
                kind, payload = item
                yield f"event: {kind}\ndata: {json.dumps(payload, default=str)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Request-ID": rid})

    # ---- answer records ---------------------------------------------------------------------
    @app.get("/api/v1/records/{request_id}")
    def record(request_id: str, channels: str | None = None, p: Principal = Auth):
        """Immutable channels of one request: owner sees user/history/control; reviewers and
        admins see raw/candidate/audit too."""
        rec = records.get(request_id)
        if rec is None:
            raise HTTPException(404, "unknown request id")
        privileged = p.is_admin or "reviewer" in p.roles
        if rec["sub"] != p.sub and not privileged:
            raise HTTPException(403, "not your request")
        allowed = {"user", "history", "control"} | ({"raw", "candidate", "audit"}
                                                      if privileged else set())
        want = [c for c in (channels or "").split(",") if c] or sorted(allowed)
        bad = [c for c in want if c not in allowed]
        if bad:
            raise HTTPException(403, f"channel(s) not available to you: {', '.join(bad)}")
        return {"request_id": request_id, "action": rec["action"], "delivered": rec["delivered"],
                "created_at": rec["created_at"], **{c: rec[c] for c in want}}  # fmt: skip

    @app.get("/api/v1/records")
    def my_records(p: Principal = Auth):
        if p.anonymous:
            raise HTTPException(401, "sign in to list your requests")
        return records.list_for(p.sub)

    # ---- chats ------------------------------------------------------------------------------
    def _chat_owner(chat_id: str, p: Principal) -> dict:
        if p.anonymous:
            raise HTTPException(401, "sign in to use chats")
        c = chats.owned(chat_id, p.sub)
        if c is None:
            raise HTTPException(404, "unknown chat")
        return c

    @app.get("/api/v1/chats")
    def chats_list(p: Principal = Auth):
        if p.anonymous:
            return []
        return chats.list_for(p.sub)

    @app.post("/api/v1/chats", status_code=201)
    def chats_create(body: schemas.ChatCreate, p: Principal = Auth):
        if p.anonymous:
            raise HTTPException(401, "sign in to use chats")
        return chats.create(p.sub, body.title or "New chat")

    @app.get("/api/v1/chats/{chat_id}")
    def chats_get(chat_id: str, p: Principal = Auth):
        c = _chat_owner(chat_id, p)
        return {**c, "messages": chats.messages(chat_id, with_response=True),
                "history": records.history_for_chat(chat_id)}  # fmt: skip

    @app.delete("/api/v1/chats/{chat_id}")
    def chats_delete(chat_id: str, p: Principal = Auth):
        _chat_owner(chat_id, p)
        chats.delete(chat_id)
        return {"deleted": chat_id}

    # ---- API keys ---------------------------------------------------------------------------
    @app.get("/api/v1/keys")
    def keys_list(p: Principal = Auth):
        if p.anonymous:
            raise HTTPException(401, "sign in to manage API keys")
        return keys.list_for(p.sub)

    @app.post("/api/v1/keys", status_code=201)
    def keys_issue(body: schemas.KeyCreate, p: Principal = Auth):
        if p.anonymous or p.via == "api_key":
            raise HTTPException(401, "sign in with a user token to issue API keys")
        k = keys.issue(p.sub, body.name)
        audit.record({"event": "key_issue", "sub": p.sub, "prefix": k["prefix"]})
        return k  # the full key is shown once

    @app.delete("/api/v1/keys/{prefix}")
    def keys_revoke(prefix: str, p: Principal = Auth):
        if p.anonymous:
            raise HTTPException(401, "sign in to manage API keys")
        if not keys.revoke(p.sub, prefix):
            raise HTTPException(404, "unknown or already revoked key")
        audit.record({"event": "key_revoke", "sub": p.sub, "prefix": prefix})
        return {"revoked": prefix}

    # ---- research tools ---------------------------------------------------------------------
    @app.post("/api/v1/tools/entities")
    def tool_entities(body: schemas.EntityToolIn, p: Principal = Auth):
        """Entities in a text, and which ones are confusable with the question's entities."""
        from app.core.symbols import neighbour_reason

        x = pipe().pool.acquire("entity_extractor")
        mentions = [m.__dict__ for m in x.mentions(body.text)] if hasattr(x, "mentions") else [
            {"surface": e, "norm": e, "kind": "?"} for e in x.extract(body.text)]
        asked = [m.__dict__ for m in x.mentions(body.question)] if (
            body.question and hasattr(x, "mentions")) else []
        confusable = []
        for m in mentions:
            for a in asked:
                r = neighbour_reason(m["norm"], a["norm"])
                if r:
                    confusable.append({"text_entity": m["surface"], "asked_entity": a["surface"],
                                       "reason": r})  # fmt: skip
        return {"mentions": mentions, "asked": asked, "confusable": confusable}

    @app.post("/api/v1/tools/verify")
    def tool_verify(body: schemas.VerifyToolIn, p: Principal = Auth):
        """Score one claim against one passage with the served verifier and entity check."""
        from app.verification.entities import entity_consistency

        pl = pipe()
        s = pl.pool.acquire("verifier").score_pairs([(body.passage, body.claim)])[0]
        checked, ok, missing = entity_consistency(body.claim, [body.passage],
                                                  pl.pool.acquire("entity_extractor"))  # fmt: skip
        return {"nli": s.model_dump(), "entities_checked": checked, "entities_ok": ok,
                "entities_missing": list(missing), "verifier": pl.versions.models["verifier"]}

    # ---- datasets ---------------------------------------------------------------------------
    @app.get("/api/v1/datasets")
    def datasets(p: Principal = Auth):
        """Split manifest summary and label counts (files only; app/ never imports evaluation/)."""
        root = Path(__file__).resolve().parents[2]
        out: dict[str, Any] = {"splits": {}, "labels": {}}
        man = root / "evaluation" / "datasets" / "splits" / "manifest_v1.json"
        if man.exists():
            m = json.loads(man.read_text())
            counts: dict[str, int] = {}
            for split in (m.get("assignments") or {}).values():
                counts[split] = counts.get(split, 0) + 1
            out["splits"] = {"version": m.get("version"), "seed": m.get("seed"),
                             "notes": m.get("notes"), "counts": counts,
                             "pinned": m.get("pinned")}  # fmt: skip
        for name, path in (("feedback", feedback_path),
                           ("review_labels", data_dir / "labels_from_review.jsonl"
                            if data_dir else None)):
            out["labels"][name] = (sum(1 for _ in open(path, encoding="utf-8"))
                                   if path and path.exists() else 0)  # fmt: skip
        return out

    # ---- admin: model profiles --------------------------------------------------------------
    @app.get("/api/v1/admin/profiles")
    def admin_profiles(p: Principal = Auth):
        require_admin(p)
        import yaml

        from models.registry.loader import DEFAULT_PATH

        raw = yaml.safe_load(Path(DEFAULT_PATH).read_text()) or {}
        return {"active": pipe().versions.profile, "default": raw.get("default_profile"),
                "profiles": {k: {"description": (v or {}).get("description", ""),
                                 "require_calibrated": (v or {}).get("require_calibrated"),
                                 "models": {r: (m if isinstance(m, str) else
                                                f"{m.get('kind')}:{m.get('model_id')}")
                                            for r, m in ((v or {}).get("models") or {}).items()}}
                             for k, v in (raw.get("profiles") or {}).items()}}  # fmt: skip

    @app.post("/api/v1/admin/profile")
    def admin_switch_profile(body: schemas.ProfileSwitch, p: Principal = Auth):
        """Build a new pipeline for the profile (pre-flight first); swap on success only."""
        require_admin(p)
        old = pipe()
        try:
            new = AnswerPipeline(old.bundle, profile=body.profile, ledger=ledger, resources=res,
                                 scope_profile=st.scope_profile, kb_root=st.kb_root)  # fmt: skip
        except Exception as e:  # noqa: BLE001  (registry/preflight errors are reported, not raised)
            raise HTTPException(409, f"cannot switch profile: {type(e).__name__}: {e}") from e
        state["pipeline"] = new
        if state["review"] is not None:
            state["review"].pipeline = new
        if state["corpus"] is not None:
            state["corpus"]._activate = new.swap_bundle
        audit.record({"event": "profile_switch", "profile": body.profile, "sub": p.sub})
        return {"active": new.versions.profile, "models": new.versions.models,
                "preflight": new.preflight}  # fmt: skip

    @app.get("/api/v1/sources/{doc_id:path}", response_model=schemas.SourceDocument)
    def source(doc_id: str, p: Principal = Auth):
        pl = pipe()
        d = pl.store.get_document(doc_id)
        if d is None and pl.kb is not None:
            d = pl.kb.store.get_document(doc_id)
        if d is None:
            raise HTTPException(404, "unknown source")
        if d.slide is not None:
            check_entitlement(p, [d.slide.course_id])
        if d.file is not None:
            readable_kbs(p, [d.file.kb_id])
        return _doc_out(d)

    return app
