"""Token budgets, usage accounting and spend caps (no model, no network).

* ``TokenCounter``: exact when a tokenizer is available, heuristic otherwise (never a hard
  dependency on tiktoken or transformers).
* ``fit_evidence``: trims the evidence list to the generator's context budget in rank order; long
  passages are cut at sentence boundaries with char offsets kept correct, so citations still point
  into the source text.
* ``Usage`` / ``Pricing`` / ``UsageLedger``: every model call records prompt/completion tokens and
  cost; the ledger enforces per-request and per-day caps and is the source for the cost columns in
  docs/COST_AND_RESOURCES.md.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, Field

from app.evidence.selector import EvidenceUnit
from app.retrieval.passages import split_sentences

_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------- counting --------
class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class HeuristicCounter:
    """~1 token per 3.6 chars for biomedical English plus one per number/symbol run. Over-estimates
    slightly on purpose: a budget that is too generous overflows the context; too tight only
    drops a low-ranked passage."""

    name = "heuristic:chars/3.6"
    _tok = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(int(len(text) / 3.6), len(self._tok.findall(text)) // 2) + 1


class HFTokenCounter:
    """Exact count with a Hugging Face tokenizer (``transformers`` optional)."""

    def __init__(self, model_id: str, revision: str | None = None):
        from transformers import AutoTokenizer  # optional dependency

        self.name = f"hf:{model_id}"
        self._tok = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def count(self, text: str) -> int:
        return len(self._tok(text, add_special_tokens=False)["input_ids"])


def make_counter(model_id: str | None = None) -> TokenCounter:
    if model_id:
        try:
            return HFTokenCounter(model_id)
        except Exception:  # noqa: BLE001  (tokenizer not available offline / not installed)
            pass
    return HeuristicCounter()


# ----------------------------------------------------------------------------- budget ----------
class GeneratorBudget(BaseModel):
    context_window: int = Field(8192, ge=512)
    reserve_output: int = Field(768, ge=64)
    prompt_overhead: int = Field(600, ge=0)
    max_evidence_tokens: int = Field(3500, ge=100)
    max_tokens_per_unit: int = Field(450, ge=50)

    @property
    def evidence_tokens(self) -> int:
        room = self.context_window - self.reserve_output - self.prompt_overhead
        return max(100, min(self.max_evidence_tokens, room))


class ResourceConfig(BaseModel):
    budgets: dict[str, dict[str, Any]] = Field(default_factory=dict)
    api: dict[str, Any] = Field(default_factory=dict)
    gpu: dict[str, Any] = Field(default_factory=dict)
    cache: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> ResourceConfig:
        p = Path(path or os.environ.get("LITBIORAG_RESOURCES_CONFIG")
                 or _ROOT / "configs" / "resources.yaml")  # fmt: skip
        return cls.model_validate(yaml.safe_load(p.read_text()) or {})

    def generator_budget(self) -> GeneratorBudget:
        return GeneratorBudget.model_validate(self.budgets.get("generator", {}))


@dataclass
class FitReport:
    budget_tokens: int
    used_tokens: int
    kept: list[str] = field(default_factory=list)
    truncated: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    counter: str = ""


def _truncate_unit(u: EvidenceUnit, max_tokens: int, counter: TokenCounter) -> EvidenceUnit | None:
    """Cut at the last sentence boundary that fits; offsets stay anchored to the source text."""
    spans = split_sentences(u.text)
    end = 0
    for _s, e in spans:
        if counter.count(u.text[:e]) > max_tokens:
            break
        end = e
    if end == 0:
        return None
    return u.model_copy(update={"text": u.text[:end].rstrip(), "char_end": u.char_start + end})


def fit_evidence(
    units: list[EvidenceUnit], budget: GeneratorBudget, counter: TokenCounter | None = None
) -> tuple[list[EvidenceUnit], FitReport]:
    """Keep evidence in rank order while it fits ``budget.evidence_tokens``."""
    counter = counter or HeuristicCounter()
    rep = FitReport(budget_tokens=budget.evidence_tokens, used_tokens=0, counter=counter.name)
    out: list[EvidenceUnit] = []
    for u in units:
        n = counter.count(u.text) + 24  # provenance header rendered per unit
        if n > budget.max_tokens_per_unit:
            cut = _truncate_unit(u, budget.max_tokens_per_unit - 24, counter)
            if cut is None:
                rep.dropped.append(u.passage_id)
                continue
            u, n = cut, counter.count(cut.text) + 24
            rep.truncated.append(u.passage_id)
        if rep.used_tokens + n > budget.evidence_tokens:
            rep.dropped.append(u.passage_id)
            continue
        out.append(u)
        rep.kept.append(u.passage_id)
        rep.used_tokens += n
    return out, rep


# ----------------------------------------------------------------------------- usage -----------
@dataclass
class Usage:
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    exact: bool = False  # True when the provider reported the counts
    cached: bool = False
    latency_ms: float = 0.0
    pricing_as_of: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d


class Pricing:
    def __init__(self, raw: dict[str, Any]):
        self.as_of: str = str(raw.get("as_of") or "")
        self.default = raw.get("default") or {"input": 0.0, "output": 0.0}
        self.models: dict[str, dict[str, float]] = raw.get("models") or {}

    @classmethod
    def load(cls, path: Path | str | None = None) -> Pricing:
        p = Path(path or _ROOT / "configs" / "pricing.yaml")
        return cls(yaml.safe_load(p.read_text()) or {}) if p.exists() else cls({})

    def rate(self, model: str) -> dict[str, float]:
        m = model.split(":", 1)[-1]  # strip 'kind:' prefix
        if m in self.models:
            return self.models[m]
        for key, r in sorted(self.models.items(), key=lambda kv: -len(kv[0])):
            if m.startswith(key):
                return r
        return self.default

    def price(self, usage: Usage) -> Usage:
        r = self.rate(usage.model)
        usage.cost_usd = (usage.prompt_tokens * float(r.get("input", 0))
                          + usage.completion_tokens * float(r.get("output", 0))) / 1e6  # fmt: skip
        usage.pricing_as_of = self.as_of
        return usage


class BudgetExceeded(RuntimeError):
    def __init__(self, scope: str, spent: float, cap: float):
        super().__init__(f"{scope} spend {spent:.4f} USD exceeds cap {cap:.4f} USD")
        self.scope, self.spent, self.cap = scope, spent, cap


class UsageLedger:
    """Append-only JSONL ledger with per-request and per-day USD caps (0 = no cap)."""

    def __init__(
        self,
        path: Path | str | None,
        *,
        max_usd_per_request: float = 0.0,
        max_usd_per_day: float = 0.0,
        pricing: Pricing | None = None,
    ):
        self.path = Path(path) if path else None
        self.max_usd_per_request = float(max_usd_per_request)
        self.max_usd_per_day = float(max_usd_per_day)
        self.pricing = pricing or Pricing({})
        self._lock = threading.Lock()
        self._day = time.strftime("%Y-%m-%d")
        self._spent_today = self._replay_today()

    def _replay_today(self) -> float:
        if not self.path or not self.path.exists():
            return 0.0
        total = 0.0
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(rec.get("ts", "")).startswith(self._day):
                    total += float(rec.get("cost_usd", 0.0))
        return total

    @property
    def spent_today(self) -> float:
        return self._spent_today

    def check_affordable(self, estimated_usd: float) -> None:
        """Raise BudgetExceeded *before* the call when a cap would be crossed."""
        if self.max_usd_per_request and estimated_usd > self.max_usd_per_request:
            raise BudgetExceeded("request", estimated_usd, self.max_usd_per_request)
        if self.max_usd_per_day and self._spent_today + estimated_usd > self.max_usd_per_day:
            raise BudgetExceeded("day", self._spent_today + estimated_usd, self.max_usd_per_day)

    def record(self, usage: Usage, *, request_id: str = "", role: str = "generator") -> Usage:
        self.pricing.price(usage)
        with self._lock:
            today = time.strftime("%Y-%m-%d")
            if today != self._day:
                self._day, self._spent_today = today, 0.0
            self._spent_today += usage.cost_usd
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "request_id": request_id,
                       "role": role, **usage.to_dict()}  # fmt: skip
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
        return usage

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        return self.pricing.price(Usage(model, prompt_tokens, completion_tokens)).cost_usd
