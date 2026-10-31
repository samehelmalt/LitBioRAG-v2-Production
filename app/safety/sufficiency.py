"""Evidence sufficiency: is there enough of the right evidence to answer at all?

Runtime-only features (no gold labels): how many distinct documents support claims, whether the
question's entities are present in the evidence, whether the exact-symbol channel found anything,
and whether BM25 and dense retrieval agree on the top documents. Score spread between ranked
scores is deliberately NOT a feature (PRODUCTION_AUDIT.md §9: it did not discriminate).

Until the calibration job fits weights (configs/gate.yaml: sufficiency_model), the heuristic
average below is used and reported as such.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from app.core.verdicts import VerificationReport
from app.retrieval.pipeline import RetrievalResult


def sufficiency_features(
    result: RetrievalResult | None, report: VerificationReport
) -> dict[str, float]:
    supporting_docs = {
        v.best_passage_id.rsplit("#", 1)[0]
        for v in report.verdicts
        if v.label == "supported" and v.best_passage_id
    }
    f: dict[str, float] = {
        "n_supporting_docs": float(min(len(supporting_docs), 5) / 5),
        "query_entity_grounding": (
            report.query_entity_grounding if report.query_entity_grounding is not None else 0.5
        ),
        "entailment_rate": report.entailment_rate if report.entailment_rate is not None else 0.0,
    }
    if result is not None:
        top = [d.doc_id for d in result.docs[:5]]
        both = sum(
            1
            for d in top
            if result.audit.rank_of(d, "bm25") is not None
            and result.audit.rank_of(d, "dense") is not None
        )
        f["bm25_dense_agreement"] = both / max(len(top), 1)
        f["entity_channel_hit"] = float(
            any(result.audit.rank_of(d, "entity") is not None for d in top)
        ) if result.audit.query_symbols else 0.5
    else:
        f["bm25_dense_agreement"] = 0.5
        f["entity_channel_hit"] = 0.5
    return f


def evidence_sufficiency(
    result: RetrievalResult | None,
    report: VerificationReport,
    model: Mapping | None = None,
) -> float:
    """Return a score in [0, 1]. ``model`` = {"weights": {feature: w}, "bias": b} when fitted."""
    f = sufficiency_features(result, report)
    if model and model.get("weights"):
        z = float(model.get("bias", 0.0)) + sum(
            float(w) * f.get(k, 0.0) for k, w in model["weights"].items()
        )
        return 1.0 / (1.0 + math.exp(-z))
    return sum(f.values()) / len(f)  # uncalibrated heuristic
