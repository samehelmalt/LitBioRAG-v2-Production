"""Generator output contract.

The generator returns JSON, not prose. The JSON is parsed into ``GeneratedAnswer``: an answer
text, a list of atomic claims (each with the citation ids it rests on and optional proposition
fields), an abstention flag, and any conflicts it saw in the evidence. Everything downstream
(verification, gate, UI) works on this object; nothing works on free text.

Parsing is tolerant of code fences and trailing chatter but never invents structure: output that
cannot be parsed is a format violation and becomes a forced abstention.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

ABSTAIN_FORMAT = "format_violation"
ABSTAIN_NO_EVIDENCE = "no_evidence"
ABSTAIN_INSUFFICIENT = "insufficient_evidence"
ABSTAIN_CONFLICT = "conflicting_evidence"
ABSTAIN_OUT_OF_SCOPE = "out_of_scope"
ABSTAIN_RESOURCE = "resource_limit"

_CIT_RE = re.compile(r"^(PMID|DOI|SLIDE):\s*(\S.*)$", re.IGNORECASE)


def normalize_citation(cit: str) -> str:
    """'pmid: 123' -> 'PMID:123'; 'doi:10.1/X' -> 'DOI:10.1/x'; unknown forms returned stripped."""
    m = _CIT_RE.match(cit.strip())
    if not m:
        return cit.strip()
    kind, val = m.group(1).upper(), m.group(2).strip()
    if kind == "PMID":
        val = "".join(ch for ch in val if ch.isdigit())
    elif kind == "DOI":
        val = val.lower()
    return f"{kind}:{val}"


class Claim(BaseModel):
    id: str = ""
    text: str
    citations: list[str] = Field(default_factory=list)
    critical: bool = Field(default=False, description="carries the answer: polarity, dose, effect")
    # proposition fields (optional; filled by the generator when it can, else by the claim parser)
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    negated: bool | None = None
    modality: str | None = Field(default=None, description="asserted | hedged | hypothetical")
    population: str | None = None
    intervention: str | None = None
    comparator: str | None = None
    outcome: str | None = None
    value: float | str | None = None
    unit: str | None = None
    time: str | None = None

    @field_validator("citations")
    @classmethod
    def _norm(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for c in v:
            n = normalize_citation(str(c))
            if n and n not in out:
                out.append(n)
        return out


class Conflict(BaseModel):
    claim_ids: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    description: str

    @field_validator("citations")
    @classmethod
    def _norm(cls, v: list[str]) -> list[str]:
        return [normalize_citation(str(c)) for c in v]


class GeneratedAnswer(BaseModel):
    answer: str = ""
    claims: list[Claim] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: str | None = None
    conflicts: list[Conflict] = Field(default_factory=list)
    format_ok: bool = True
    unknown_citations: list[str] = Field(
        default_factory=list, description="cited ids that are not in the evidence set"
    )
    raw: str = ""
    prompt_id: str = ""
    model: str = ""
    question: str = ""
    usage: dict | None = Field(default=None, description="token/cost accounting for this call")
    evidence_fit: dict | None = Field(default=None, description="what the token budget kept/cut")

    def all_citations(self) -> list[str]:
        seen: list[str] = []
        for c in self.claims:
            for cit in c.citations:
                if cit not in seen:
                    seen.append(cit)
        return seen

    @staticmethod
    def abstention(reason: str, *, raw: str = "", prompt_id: str = "", model: str = "",
                   text: str | None = None, format_ok: bool = True) -> GeneratedAnswer:  # fmt: skip
        return GeneratedAnswer(
            answer=text or ABSTENTION_TEXT[reason],
            abstained=True,
            abstain_reason=reason,
            format_ok=format_ok,
            raw=raw,
            prompt_id=prompt_id,
            model=model,
        )


ABSTENTION_TEXT = {
    ABSTAIN_NO_EVIDENCE: "No relevant evidence was retrieved for this question.",
    ABSTAIN_INSUFFICIENT: (
        "The available biomedical evidence is insufficient to answer this question reliably."
    ),
    ABSTAIN_CONFLICT: (
        "The retrieved literature contains conflicting evidence, so a definitive conclusion "
        "cannot be established."
    ),
    ABSTAIN_FORMAT: "The system could not produce a verifiable answer for this question.",
    ABSTAIN_RESOURCE: (
        "The system paused answering because a resource limit (API spend or model availability) "
        "was reached. No answer was generated."
    ),
    ABSTAIN_OUT_OF_SCOPE: (
        "This question is outside the scope of literature support that this system provides."
    ),
}

# The schema the generator is asked to follow (kept small and literal for small models).
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["answer", "claims", "abstained"],
    "properties": {
        "answer": {"type": "string", "description": "concise answer built only from the claims"},
        "abstained": {"type": "boolean"},
        "abstain_reason": {
            "type": ["string", "null"],
            "enum": [None, ABSTAIN_INSUFFICIENT, ABSTAIN_CONFLICT, ABSTAIN_OUT_OF_SCOPE],
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "citations"],
                "properties": {
                    "text": {"type": "string", "description": "one atomic factual statement"},
                    "citations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "ids copied exactly from the evidence list, e.g. PMID:123",
                    },
                    "critical": {"type": "boolean"},
                    "negated": {"type": "boolean"},
                    "population": {"type": ["string", "null"]},
                    "value": {"type": ["number", "string", "null"]},
                    "unit": {"type": ["string", "null"]},
                },
            },
        },
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["description"],
                "properties": {
                    "description": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


def schema_text() -> str:
    return json.dumps(ANSWER_SCHEMA, indent=1)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(raw: str) -> str | None:
    """Return the most plausible JSON object substring of a model output, or None."""
    if not raw:
        return None
    m = _FENCE_RE.search(raw)
    if m:
        raw = m.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return raw[start : end + 1]


def parse_answer(raw: str, *, prompt_id: str = "", model: str = "") -> GeneratedAnswer:
    """Parse generator output. Any failure to produce the contract is a forced abstention."""
    blob = extract_json_object(raw)
    if blob is None:
        return GeneratedAnswer.abstention(ABSTAIN_FORMAT, raw=raw, prompt_id=prompt_id,
                                          model=model, format_ok=False)  # fmt: skip
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        try:  # common small-model slip: trailing commas
            data = json.loads(re.sub(r",\s*([}\]])", r"\1", blob))
        except json.JSONDecodeError:
            return GeneratedAnswer.abstention(ABSTAIN_FORMAT, raw=raw, prompt_id=prompt_id,
                                              model=model, format_ok=False)  # fmt: skip
    if not isinstance(data, dict):
        return GeneratedAnswer.abstention(ABSTAIN_FORMAT, raw=raw, prompt_id=prompt_id,
                                          model=model, format_ok=False)  # fmt: skip
    claims_in = data.get("claims") or []
    claims: list[Claim] = []
    for i, c in enumerate(claims_in, start=1):
        if isinstance(c, str):
            c = {"text": c, "citations": []}
        if not isinstance(c, dict) or not str(c.get("text", "")).strip():
            continue
        c = {k: v for k, v in c.items() if k in Claim.model_fields}
        c.setdefault("citations", [])
        if not isinstance(c["citations"], list):
            c["citations"] = [c["citations"]]
        claims.append(Claim(id=f"C{i}", **c))
    conflicts = []
    for k in data.get("conflicts") or []:
        if isinstance(k, dict) and k.get("description"):
            conflicts.append(Conflict(**{x: y for x, y in k.items() if x in Conflict.model_fields}))
    abstained = bool(data.get("abstained", False)) or not claims
    reason = data.get("abstain_reason") if abstained else None
    if abstained and reason not in ABSTENTION_TEXT:
        reason = ABSTAIN_INSUFFICIENT
    answer = str(data.get("answer") or "").strip()
    if abstained and not answer:
        answer = ABSTENTION_TEXT[reason]
    return GeneratedAnswer(
        answer=answer,
        claims=claims if not abstained else claims,
        abstained=abstained,
        abstain_reason=reason,
        conflicts=conflicts,
        format_ok=True,
        raw=raw,
        prompt_id=prompt_id,
        model=model,
    )
