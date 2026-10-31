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

from app.evidence.selector import EvidenceUnit
from app.generation.base import Generator
from app.generation.contract import ABSTAIN_NO_EVIDENCE, GeneratedAnswer, parse_answer
from app.generation.prompts import PROMPT_ID, build_messages


class GenerationConfig(BaseModel):
    max_tokens: int = Field(1024, ge=64)
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_evidence_units: int = Field(10, ge=1)

    @classmethod
    def from_yaml(cls, path: Path | str) -> GenerationConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})


def answer(
    question: str,
    evidence: Sequence[EvidenceUnit],
    generator: Generator,
    *,
    config: GenerationConfig | None = None,
) -> GeneratedAnswer:
    cfg = config or GenerationConfig()
    units = list(evidence)[: cfg.max_evidence_units]
    if not units:
        out = GeneratedAnswer.abstention(ABSTAIN_NO_EVIDENCE, prompt_id=PROMPT_ID,
                                         model=generator.name)  # fmt: skip
        out.question = question
        return out
    messages = build_messages(question, units)
    raw = generator.generate(
        messages, max_tokens=cfg.max_tokens, temperature=cfg.temperature, evidence=units
    )
    parsed = parse_answer(raw, prompt_id=PROMPT_ID, model=generator.name)
    known = {u.citation_id for u in units}
    parsed.unknown_citations = [c for c in parsed.all_citations() if c not in known]
    parsed.question = question
    return parsed
