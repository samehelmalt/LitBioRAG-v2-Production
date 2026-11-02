"""Answer generation service: evidence in, GeneratedAnswer out.

Guarantees independent of the model behind the ``Generator`` protocol:
* no evidence → abstention without a model call;
* every output passes through the contract parser (unparseable → forced abstention);
* every cited id is checked against the evidence set; ids not present are recorded in
  ``unknown_citations`` so the verifier fails them (fabricated citations never pass silently).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.core.budget import (
    BudgetExceeded,
    GeneratorBudget,
    HeuristicCounter,
    TokenCounter,
    Usage,
    UsageLedger,
    fit_evidence,
)
from app.evidence.compress import compress_evidence
from app.evidence.selector import EvidenceUnit
from app.generation.base import Generator
from app.generation.contract import (
    ABSTAIN_NO_EVIDENCE,
    ABSTAIN_RESOURCE,
    GeneratedAnswer,
    parse_answer,
)
from app.generation.prompts import PROMPT_ID, build_messages


class GenerationConfig(BaseModel):
    max_tokens: int = Field(1024, ge=64)
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_evidence_units: int = Field(10, ge=1)
    compress_evidence: bool = True
    max_sentences_per_unit: int = Field(4, ge=1)
    cache_generations: bool = True
    max_attempts: int = Field(2, ge=1, le=4)  # bounded regeneration with precise feedback
    profiles: list[dict] = Field(default_factory=list)  # overrides for app/generation/profiles

    @classmethod
    def from_yaml(cls, path: Path | str) -> GenerationConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})


def _abstain(reason: str, question: str, generator: Generator, **extra) -> GeneratedAnswer:
    out = GeneratedAnswer.abstention(reason, prompt_id=PROMPT_ID, model=generator.name)
    out.question = question
    for k, v in extra.items():
        setattr(out, k, v)
    return out


def answer(
    question: str,
    evidence: Sequence[EvidenceUnit],
    generator: Generator,
    *,
    config: GenerationConfig | None = None,
    budget: GeneratorBudget | None = None,
    counter: TokenCounter | None = None,
    ledger: UsageLedger | None = None,
    request_id: str = "",
    history: list[str] | None = None,
    cache=None,
    feedback: str | None = None,
    lecture_first: bool = False,
) -> GeneratedAnswer:
    """Evidence → prompt → model → parsed answer, under a token budget and a spend cap.

    * ``budget`` trims evidence to the generator's context (rank order; sentence-boundary cuts);
    * ``ledger`` is asked *before* the call whether the estimated cost is affordable, and records
      the real usage afterwards. A cap that would be crossed yields an abstention with reason
      ``resource_limit`` — never a silent failure, never a partial answer.
    """
    cfg = config or GenerationConfig()
    counter = counter or HeuristicCounter()
    units = list(evidence)[: cfg.max_evidence_units]
    if not units:
        return _abstain(ABSTAIN_NO_EVIDENCE, question, generator)
    full_tokens = sum(counter.count(u.text) for u in units)
    if cfg.compress_evidence:
        units = compress_evidence(units, question, max_sentences=cfg.max_sentences_per_unit)
    fit = None
    if budget is not None:
        units, rep = fit_evidence(units, budget, counter)
        fit = rep.__dict__
        if not units:
            return _abstain(ABSTAIN_NO_EVIDENCE, question, generator, evidence_fit=fit)
    fit = fit or {}
    fit["evidence_tokens_full"] = full_tokens
    fit["evidence_tokens_sent"] = sum(counter.count(u.text) for u in units)
    messages = build_messages(question, units, history, lecture_first=lecture_first)
    if feedback:
        messages = [*messages, {"role": "user", "content": feedback}]
    prompt_tokens = sum(counter.count(m.get("content", "")) for m in messages)
    cache_key = None
    if cache is not None and cfg.cache_generations and not feedback:
        import hashlib

        h = hashlib.sha256()
        for part in (generator.name, PROMPT_ID, question.strip().lower(),
                     *(f"{u.passage_id}:{u.char_start}:{u.char_end}" for u in units),
                     *(history or [])):
            h.update(part.encode("utf-8"))
            h.update(b"\x1f")
        cache_key = h.hexdigest()
        hit = cache.get("gen", cache_key)
        if hit is not None:
            parsed = parse_answer(hit["raw"], prompt_id=PROMPT_ID, model=generator.name)
            known = {u.citation_id for u in units}
            parsed.unknown_citations = [c for c in parsed.all_citations() if c not in known]
            parsed.question = question
            parsed.usage = {**hit["usage"], "cached": True, "cost_usd": 0.0}
            parsed.evidence_fit = fit
            return parsed
    if ledger is not None:
        try:
            ledger.check_affordable(
                ledger.estimate(generator.name, prompt_tokens, cfg.max_tokens)
            )
        except BudgetExceeded as e:
            usage = Usage(model=generator.name, prompt_tokens=prompt_tokens)
            blocked = {**usage.to_dict(), "blocked_by": str(e)}
            return _abstain(ABSTAIN_RESOURCE, question, generator, usage=blocked,
                            evidence_fit=fit)  # fmt: skip
    raw = generator.generate(
        messages, max_tokens=cfg.max_tokens, temperature=cfg.temperature, evidence=units
    )
    usage = getattr(generator, "last_usage", None) or Usage(
        model=generator.name, prompt_tokens=prompt_tokens, completion_tokens=counter.count(raw)
    )
    if ledger is not None:
        ledger.record(usage, request_id=request_id, role="generator")
    if cache_key is not None:
        cache.put("gen", cache_key, {"raw": raw, "usage": usage.to_dict()})
    parsed = parse_answer(raw, prompt_id=PROMPT_ID, model=generator.name)
    known = {u.citation_id for u in units}
    parsed.unknown_citations = [c for c in parsed.all_citations() if c not in known]
    parsed.question = question
    parsed.usage = usage.to_dict()
    parsed.evidence_fit = fit
    return parsed
