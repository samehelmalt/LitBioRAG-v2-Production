"""Fit the gate's confidence and sufficiency models and pick thresholds on the calibration split.

    python -m scripts.calibrate_gate --rows evaluation/benchmarks/runs/<run>/rows.jsonl \
        --out configs/gate.yaml --run-id <run>

Inputs are rows written by evaluation/runners/answer.py (features + proxy or human labels).
Fits two logistic regressions by plain gradient descent (no sklearn): confidence(features) →
P(acceptable) and sufficiency(sufficiency_features) → P(acceptable). Chooses
``confidence_answer`` by minimizing expected cost over the rows with the cost weights in the gate
config, and writes the result into the YAML with ``calibrated_on`` set. Emits a JSON report with
the risk–coverage curve, ECE and Brier next to the rows. Never run on the test split.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics.trust import brier, coverage_risk_curve, ece  # noqa: E402


def fit_logistic(X: list[dict[str, float]], y: list[bool], *, steps: int = 2000, lr: float = 0.1,
                 l2: float = 1e-3) -> dict:  # fmt: skip
    keys = sorted({k for row in X for k in row})
    w = dict.fromkeys(keys, 0.0)
    b = 0.0
    n = len(X)
    for _ in range(steps):
        gw = dict.fromkeys(keys, 0.0)
        gb = 0.0
        for row, label in zip(X, y, strict=True):
            z = b + sum(w[k] * row.get(k, 0.0) for k in keys)
            p = 1.0 / (1.0 + math.exp(-z))
            err = p - float(label)
            for k in keys:
                gw[k] += err * row.get(k, 0.0)
            gb += err
        for k in keys:
            w[k] -= lr * (gw[k] / n + l2 * w[k])
        b -= lr * gb / n
    return {"weights": w, "bias": b, "features": keys}


def predict(model: dict, row: dict[str, float]) -> float:
    z = model["bias"] + sum(model["weights"][k] * row.get(k, 0.0) for k in model["weights"])
    return 1.0 / (1.0 + math.exp(-z))


def choose_threshold(conf: list[float], labels: list[bool], costs: dict[str, float]) -> float:
    """Threshold on confidence that minimizes expected cost (ANSWER above, CAUTION below)."""
    best_t, best_cost = 0.7, float("inf")
    for t in sorted(set(conf)) + [1.01]:
        cost = 0.0
        for c, ok in zip(conf, labels, strict=True):
            if c >= t:
                cost += 0.0 if ok else costs["false_pass"]
            else:
                cost += costs["warn"] + (0.0 if ok else costs["warn"])
        if cost < best_cost:
            best_cost, best_t = cost, t
    return min(max(best_t, 0.0), 1.0)


def choose_threshold_for_precision(conf: list[float], labels: list[bool], target: float,
                                   *, min_released: int = 10) -> tuple[float, dict]:  # fmt: skip
    """Lowest confidence threshold whose released answers reach ``target`` precision on the
    calibration rows (maximum coverage at that precision). Returns (threshold, point) where point
    reports the achieved precision and coverage; threshold 1.01 (release nothing automatically)
    when the target is unreachable — then every answer goes to ANSWER_WITH_CAUTION / review."""
    order = sorted(zip(conf, labels, strict=True), key=lambda t: -t[0])
    n = len(order)
    best = (1.01, {"precision": None, "coverage": 0.0, "released": 0, "target": target})
    ok = 0
    for i, (c, a) in enumerate(order, start=1):
        ok += 1 if a else 0
        if i < min_released or (i < n and order[i][0] == c):
            continue
        precision = ok / i
        if precision >= target:
            best = (c, {"precision": precision, "coverage": i / n, "released": i,
                        "target": target})  # fmt: skip
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "configs" / "gate.yaml")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--min-rows", type=int, default=50)
    ap.add_argument("--target-precision", type=float, default=None,
                    help="operating point: release automatically only above the confidence at "
                         "which released answers reach this precision (e.g. 0.95); default: "
                         "expected-cost threshold")
    a = ap.parse_args(argv)

    rows = [json.loads(line) for line in a.rows.read_text().splitlines() if line.strip()]
    if len(rows) < a.min_rows:
        print(f"refusing to calibrate on {len(rows)} rows (< {a.min_rows})", file=sys.stderr)
        return 2
    labels = [bool(r["proxy_acceptable"]) for r in rows]
    cfg = yaml.safe_load(a.out.read_text()) if a.out.exists() else {}
    default_costs = {"false_pass": 10.0, "false_block": 1.0, "warn": 0.3, "abstain": 1.0}
    costs = cfg.get("costs") or default_costs

    conf_model = fit_logistic([r["features"] for r in rows], labels)
    suff_model = fit_logistic([r["sufficiency_features"] for r in rows], labels)
    conf = [predict(conf_model, r["features"]) for r in rows]
    operating_point: dict = {"mode": "expected_cost"}
    if a.target_precision is not None:
        t, point = choose_threshold_for_precision(conf, labels, a.target_precision)
        operating_point = {"mode": "target_precision", **point}
    else:
        t = choose_threshold(conf, labels, costs)

    cfg.update(
        {
            "calibrated_on": a.run_id,
            "confidence_answer": round(t, 4),
            "confidence_model": conf_model,
            "sufficiency_model": suff_model,
            "costs": costs,
            "operating_point": operating_point,
        }
    )
    a.out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    report = {
        "run_id": a.run_id,
        "n_rows": len(rows),
        "labels": "proxy" if all("proxy_acceptable" in r for r in rows) else "human",
        "confidence_answer": t,
        "operating_point": operating_point,
        "ece": ece(conf, labels),
        "brier": brier(conf, labels),
        "coverage_risk": coverage_risk_curve(conf, labels),
    }
    (a.rows.parent / "calibration.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "coverage_risk"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
