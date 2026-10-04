"""Labels for trust metrics.

Until human faithfulness labels exist (owner decision 2026-10-04: proxy labels only for now),
an answer is *acceptable to release* when every available proxy agrees:

* if the question is yes/no, the predicted polarity equals the gold polarity;
* at least one gold document was retrieved into the evidence set;
* the unsupported-claim rate is at most ``MAX_UNSUPPORTED``;
* no claim is contradicted.

These are necessary, not sufficient, conditions for a faithful answer. Trust metrics computed
on proxy labels over-estimate acceptability and must be reported as such (see docs/EVALUATION.md).

The same module defines the JSONL record used to export answers for human labelling and to
import the ratings back, so that the switch from proxy to human labels is a one-line change.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

MAX_UNSUPPORTED = 0.25
LABEL_VALUES = ("acceptable", "unacceptable", "unsure")


def proxy_acceptable(
    *,
    yn_gold: str | None,
    yn_pred: str | None,
    gold_retrieved: bool,
    unsupported_rate: float | None,
    contradiction_rate: float | None,
    max_unsupported: float = MAX_UNSUPPORTED,
) -> bool:
    """Proxy label. Missing rates count as failures (we cannot show the answer is acceptable)."""
    if yn_gold:
        if not yn_pred or yn_pred.lower() != yn_gold.lower():
            return False
    if not gold_retrieved:
        return False
    if unsupported_rate is None or unsupported_rate > max_unsupported:
        return False
    if contradiction_rate is None or contradiction_rate > 0.0:
        return False
    return True


@dataclass(frozen=True)
class LabelRecord:
    query_id: str
    question: str
    answer: str
    evidence: list[dict] = field(default_factory=list)  # [{id, title, year, source_type, text}]
    label: str = ""  # one of LABEL_VALUES once rated
    rater: str = ""
    notes: str = ""
    run_id: str = ""
    proxy_label: bool | None = None

    def __post_init__(self) -> None:
        if self.label and self.label not in LABEL_VALUES:
            raise ValueError(f"label must be one of {LABEL_VALUES}")


def write_jsonl(records: Iterable[LabelRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[LabelRecord]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield LabelRecord(**json.loads(line))
