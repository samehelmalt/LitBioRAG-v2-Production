"""One object that runs retrieve → evidence → generate → verify → gate for a question.

Shared by the API, the demo script and the evaluation runners so there is exactly one code path
from question to delivery decision. Models come from the registry profile; the generator goes
through ``ModelPool`` so an ``on_demand`` local generator is loaded per request and freed after.
"""

from __future__ import annotations

import json
import time
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
    ):
        self.scope_profile = scope_profile
        self.bundle = Path(bundle)
        self.registry = ModelRegistry.load(profile=profile)
        self.resources = resources or ResourceConfig.from_yaml()
        self.pool = ModelPool(self.registry, self.resources)
        self.preflight = self.pool.preflight()
        self.store = MetadataStore(self.bundle / "metadata.sqlite", read_only=True)
        self.retriever = HybridRetriever(
            self.store,
            BM25Index(self.bundle / "bm25"),
            DenseIndex(self.bundle / "dense", self.pool.acquire("dense_encoder")),
            self.pool.acquire("reranker"),
            RetrievalConfig.from_yaml(configs / "retrieval.yaml"),
        )
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

    def run(
        self,
        question: str,
        *,
        course_ids: list[str] | None = None,
        request_id: str = "",
        max_units: int | None = None,
    ) -> PipelineOutput:
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        scope = classify_scope(question, self.scope_profile)
        if not scope.in_scope:
            # out of scope: no retrieval, no model call, no tokens spent
            gen = GeneratedAnswer.abstention(ABSTAIN_OUT_OF_SCOPE, prompt_id=PROMPT_ID,
                                             model="none")  # fmt: skip
            gen.question = question
            report = VerificationReport()
            decision = decide(report, gen, config=self.gate_cfg, scope=scope, question=question)
            t["total"] = (time.perf_counter() - t0) * 1000
            return PipelineOutput(question, None, [], gen, report, decision,
                                  {k: round(v, 1) for k, v in t.items()})  # fmt: skip
        res = self.retriever.retrieve(question, mode="hybrid+rerank", course_ids=course_ids)
        units = select_evidence(res, self.store, max_units=max_units or self.max_units)
        t["retrieval"] = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        with self.pool.use("generator") as generator:
            gen = generate_answer(
                question, units, generator, config=self.gen_cfg, budget=self.budget,
                ledger=self.ledger, request_id=request_id,
            )  # fmt: skip
        t["generation"] = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        report = verify(
            gen, units, self.pool.acquire("verifier"), self.pool.acquire("entity_extractor"),
            config=self.ver_cfg, thresholds=self.registry.thresholds("verifier"),
            question=question,
        )  # fmt: skip
        suff = evidence_sufficiency(res, report, self.gate_cfg.sufficiency_model)
        decision = decide(report, gen, sufficiency=suff, config=self.gate_cfg, question=question)
        t["verification"] = (time.perf_counter() - t2) * 1000
        t["total"] = (time.perf_counter() - t0) * 1000
        return PipelineOutput(question, res, units, gen, report, decision,
                              {k: round(v, 1) for k, v in t.items()})  # fmt: skip
