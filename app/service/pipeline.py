"""One object that runs retrieve → evidence → generate → verify → gate for a question.

Shared by the API, the demo script and the evaluation runners so there is exactly one code path
from question to delivery decision. Models come from the registry profile; the generator goes
through ``ModelPool`` so an ``on_demand`` local generator is loaded per request and freed after.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.budget import ResourceConfig, UsageLedger
from app.core.verdicts import VerificationReport
from app.evidence.selector import EvidenceUnit, select_evidence
from app.generation.contract import ABSTAIN_OUT_OF_SCOPE, GeneratedAnswer
from app.generation.prompts import PROMPT_ID
from app.generation.service import GenerationConfig
from app.generation.service import answer as generate_answer
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.multi import SCOPES, MultiRetriever
from app.retrieval.pipeline import HybridRetriever, RetrievalConfig, RetrievalResult
from app.retrieval.store import MetadataStore
from app.safety.gate import Decision, GateConfig, decide
from app.safety.scope import classify_scope
from app.safety.sufficiency import evidence_sufficiency
from app.verification.pipeline import VerificationConfig, verify
from models.registry import ModelRegistry
from models.registry.pool import ModelPool

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"


@dataclass
class PipelineOutput:
    question: str
    retrieval: RetrievalResult | None  # None when scope stopped the request before retrieval
    evidence: list[EvidenceUnit]
    generated: GeneratedAnswer
    report: VerificationReport
    decision: Decision
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Versions:
    profile: str
    models: dict[str, dict]
    bundle_id: str
    bundle_created_at: str
    prompt_id: str
    gate_calibrated_on: str
    api_version: str = "v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile, "models": self.models, "bundle_id": self.bundle_id,
            "bundle_created_at": self.bundle_created_at, "prompt_id": self.prompt_id,
            "gate_calibrated_on": self.gate_calibrated_on or "UNCALIBRATED",
            "api_version": self.api_version,
        }  # fmt: skip


class AnswerPipeline:
    def __init__(
        self,
        bundle: Path,
        *,
        profile: str | None = None,
        configs: Path = CONFIGS,
        ledger: UsageLedger | None = None,
        resources: ResourceConfig | None = None,
        max_units: int = 8,
        scope_profile: str = "research",
        kb_root: Path | None = None,
    ):
        self.scope_profile = scope_profile
        self.bundle = Path(bundle)
        self.registry = ModelRegistry.load(profile=profile)
        self.resources = resources or ResourceConfig.from_yaml()
        self.pool = ModelPool(self.registry, self.resources)
        self.preflight = self.pool.preflight()
        self.store = MetadataStore(self.bundle / "metadata.sqlite", read_only=True)
        ret_cfg = RetrievalConfig.from_yaml(configs / "retrieval.yaml")
        encoder = self.pool.acquire("dense_encoder")
        self.retriever = HybridRetriever(
            self.store,
            BM25Index(self.bundle / "bm25"),
            DenseIndex(self.bundle / "dense", encoder),
            self.pool.acquire("reranker"),
            ret_cfg,
        )
        # uploaded knowledge bases: a second, incrementally indexed source (app/kb)
        self.kb = None
        if kb_root is not None:
            from app.kb.indexer import KBIndexer

            self.kb = KBIndexer(kb_root, encoder)
        self.ret_cfg = ret_cfg
        self.gen_cfg = GenerationConfig.from_yaml(configs / "generation.yaml")
        self.ver_cfg = VerificationConfig.from_yaml(configs / "verification.yaml")
        self.gate_cfg = GateConfig.from_yaml(configs / "gate.yaml")
        self.budget = self.resources.generator_budget()
        self.ledger = ledger
        self.max_units = max_units
        manifest: dict[str, Any] = {}
        mpath = self.bundle / "manifest.json"
        if mpath.exists():
            manifest = json.loads(mpath.read_text())
        self.versions = Versions(
            profile=self.registry.profile.name, models=self.registry.describe(),
            bundle_id=str(manifest.get("corpus_id", self.bundle.name)),
            bundle_created_at=str(manifest.get("created_at", "")),
            prompt_id=PROMPT_ID, gate_calibrated_on=self.gate_cfg.calibrated_on or "",
        )  # fmt: skip

    def swap_bundle(self, bundle: Path) -> dict:
        """Point retrieval at another verified bundle without reloading models."""
        bundle = Path(bundle)
        encoder = self.pool.acquire("dense_encoder")
        store = MetadataStore(bundle / "metadata.sqlite", read_only=True)
        retriever = HybridRetriever(store, BM25Index(bundle / "bm25"),
                                    DenseIndex(bundle / "dense", encoder),
                                    self.pool.acquire("reranker"), self.ret_cfg)  # fmt: skip
        manifest: dict = {}
        if (bundle / "manifest.json").exists():
            manifest = json.loads((bundle / "manifest.json").read_text())
        self.bundle, self.store, self.retriever = bundle, store, retriever
        self.versions = Versions(
            profile=self.versions.profile, models=self.versions.models,
            bundle_id=str(manifest.get("corpus_id", bundle.name)),
            bundle_created_at=str(manifest.get("created_at", "")),
            prompt_id=self.versions.prompt_id, gate_calibrated_on=self.versions.gate_calibrated_on,
        )  # fmt: skip
        return {"bundle_id": self.versions.bundle_id, "counts": store.counts()}

    def multi(self) -> MultiRetriever:
        """Fresh per call: the uploads dense index is rebuilt after every upload."""
        uploads = None
        if self.kb is not None and self.kb.has_content():
            uploads = HybridRetriever(self.kb.store, self.kb.bm25, self.kb.dense, None,
                                      self.ret_cfg)  # fmt: skip
        return MultiRetriever(self.retriever, uploads, self.pool.acquire("reranker"),
                              self.ret_cfg)  # fmt: skip

    def run(
        self,
        question: str,
        *,
        course_ids: list[str] | None = None,
        request_id: str = "",
        max_units: int | None = None,
        scope: str = "indexed",
        kb_ids: list[str] | None = None,
        history: list[str] | None = None,
        progress: Callable[[str, dict], None] | None = None,
    ) -> PipelineOutput:
        """``scope``: indexed (served corpus bundle) | uploaded (the caller's knowledge bases
        only) | both. ``kb_ids``: knowledge bases the caller may read (entitlement)."""
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")
        t: dict[str, float] = {}
        t0 = time.perf_counter()

        def say(stage: str, **info) -> None:
            if progress is not None:
                progress(stage, info)

        say("scope_check")
        scope_dec = classify_scope(question, self.scope_profile)
        if not scope_dec.in_scope:
            # out of scope: no retrieval, no model call, no tokens spent
            gen = GeneratedAnswer.abstention(ABSTAIN_OUT_OF_SCOPE, prompt_id=PROMPT_ID,
                                             model="none")  # fmt: skip
            gen.question = question
            report = VerificationReport()
            decision = decide(report, gen, config=self.gate_cfg, scope=scope_dec,
                              question=question)  # fmt: skip
            t["total"] = (time.perf_counter() - t0) * 1000
            return PipelineOutput(question, None, [], gen, report, decision,
                                  {k: round(v, 1) for k, v in t.items()})  # fmt: skip
        say("retrieving", scope=scope)
        if scope == "indexed":
            res = self.retriever.retrieve(question, mode="hybrid+rerank", course_ids=course_ids)
            store = self.store
        else:
            m = self.multi()
            res = m.retrieve(question, scope=scope, course_ids=course_ids, kb_ids=kb_ids)
            store = m.store
        units = select_evidence(res, store, max_units=max_units or self.max_units)
        t["retrieval"] = (time.perf_counter() - t0) * 1000
        say("retrieved", documents=len(res.docs), evidence_units=len(units),
            ms=round(t["retrieval"], 1))  # fmt: skip
        t1 = time.perf_counter()
        say("generating")
        with self.pool.use("generator") as generator:
            gen = generate_answer(
                question, units, generator, config=self.gen_cfg, budget=self.budget,
                ledger=self.ledger, request_id=request_id, history=history,
            )  # fmt: skip
        t["generation"] = (time.perf_counter() - t1) * 1000
        say("verifying", claims=len(gen.claims))
        t2 = time.perf_counter()
        report = verify(
            gen, units, self.pool.acquire("verifier"), self.pool.acquire("entity_extractor"),
            config=self.ver_cfg, thresholds=self.registry.thresholds("verifier"),
            question=question,
        )  # fmt: skip
        say("deciding")
        suff = evidence_sufficiency(res, report, self.gate_cfg.sufficiency_model)
        decision = decide(report, gen, sufficiency=suff, config=self.gate_cfg, question=question)
        t["verification"] = (time.perf_counter() - t2) * 1000
        t["total"] = (time.perf_counter() - t0) * 1000
        return PipelineOutput(question, res, units, gen, report, decision,
                              {k: round(v, 1) for k, v in t.items()})  # fmt: skip
