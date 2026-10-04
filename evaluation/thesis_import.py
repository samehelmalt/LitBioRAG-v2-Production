"""Load the thesis master-run CSVs and reproduce the numbers in docs/THESIS_BASELINE.md.

Usage:
    python -m evaluation.thesis_import --report

Nothing here re-runs any model. It reads ``evaluation/benchmarks/thesis_baseline/V5521_Run`` and
recomputes aggregates, which the regression test compares with the documented values. It also
re-implements the thesis gate policy from the six logged signal columns, so that later
experiments (for example re-scoring Signal_NLI without the gold label) can ask how decisions
would have changed.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from evaluation.labels import proxy_acceptable
from evaluation.metrics.trust import gate_confusion

MASTER_DIR = Path(__file__).resolve().parent / "benchmarks" / "thesis_baseline" / "V5521_Run"
STEPS = ["S-1", "S0", "S1", "S2", "S3", "S3a", "S4", "S5v1", "S5", "S6", "S7"]

THESIS_WEIGHTS = {
    "Signal_NLI": 0.30,
    "Signal_Numeric": 0.20,
    "Signal_Entity": 0.10,
    "Signal_Retrieval": 0.25,
    "not_contradiction": 0.10,
    "not_unsupported": 0.05,
}
THESIS_THRESHOLDS = {"pass": 0.28, "warn": 0.18}
BLOCK_POST_GATE_CONSTANT = 0.90  # value the thesis assigned to blocked answers (engine L6820)


def load_master(directory: Path = MASTER_DIR) -> pd.DataFrame:
    files = sorted(glob.glob(str(directory / "thesis_ladder_*.csv")))
    if not files:
        raise FileNotFoundError(f"no thesis_ladder_*.csv under {directory}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        tag = pd.DataFrame({"source_file": [Path(f).name] * len(df)}, index=df.index)
        frames.append(pd.concat([df, tag], axis=1))
    return pd.concat(frames, ignore_index=True)


def ladder_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Step").agg(
        recall_k_norm=("Recall_K_Norm", "mean"),
        mrr_norm=("MRR_norm", "mean"),
        nli_faithfulness=("Faithfulness", "mean"),
        latency_median_s=("latency_s", "median"),
        n=("Step", "size"),
    )
    return g.loc[[s for s in STEPS if s in g.index]]


def s7(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Step"] == "S7"].copy()


def gate_summary(df: pd.DataFrame) -> dict:
    x = s7(df)
    counts = x["Gate"].value_counts().to_dict()
    return {
        "n": int(len(x)),
        "counts": {k: int(v) for k, v in counts.items()},
        "pre_gate_mean": float(x["Pre_Gate_Faith"].mean()),
        "post_gate_mean": float(x["Post_Gate_Faith"].mean()),
        "delta_mean": float((x["Post_Gate_Faith"] - x["Pre_Gate_Faith"]).mean()),
        "block_post_gate_values": sorted(
            x.loc[x["Gate"] == "BLOCK", "Post_Gate_Faith"].round(4).unique().tolist()
        ),
    }


def proxy_labels_s7(df: pd.DataFrame, max_unsupported: float = 0.25) -> pd.Series:
    """Proxy acceptability per S7 row from the logged columns (see evaluation/labels.py)."""
    x = s7(df)
    out = []
    for _, r in x.iterrows():
        yn_gold = r["YN_Label_GT"] if isinstance(r["YN_Label_GT"], str) else None
        yn_pred = r["YN_Predicted"] if isinstance(r["YN_Predicted"], str) else None
        out.append(
            proxy_acceptable(
                yn_gold=yn_gold or None,
                yn_pred=yn_pred,
                gold_retrieved=float(r["Recall_K_Norm"]) > 0,
                unsupported_rate=float(r["Unsupported"]),
                contradiction_rate=float(r["Contradiction"]),
                max_unsupported=max_unsupported,
            )
        )
    return pd.Series(out, index=x.index, name="proxy_acceptable")


def false_pass_proxies(df: pd.DataFrame) -> dict:
    x = s7(df)
    p = x[x["Gate"] == "PASS"]
    yn = p[p["YN_Correctness"] >= 0]
    b = x[x["Gate"] == "BLOCK"]
    byn = b[b["YN_Correctness"] >= 0]
    conf = gate_confusion(x["Gate"].tolist(), proxy_labels_s7(df).tolist())
    return {
        "pass_rows": int(len(p)),
        "pass_yesno_rows": int(len(yn)),
        "pass_yesno_wrong": int((yn["YN_Correctness"] == 0).sum()),
        "pass_gold_not_retrieved": int((p["Recall_K_Norm"] == 0).sum()),
        "pass_mean_unsupported": float(p["Unsupported"].mean()),
        "block_yesno_rows": int(len(byn)),
        "block_yesno_correct": int((byn["YN_Correctness"] == 1).sum()),
        "proxy_false_pass_rate": conf.false_pass_rate,
        "proxy_false_warn_rate": conf.false_warn_rate,
        "proxy_false_block_rate": conf.false_block_rate,
        "proxy_coverage": conf.coverage,
    }


def recompute_gate(
    df: pd.DataFrame,
    weights: dict[str, float] = THESIS_WEIGHTS,
    thresholds: dict[str, float] = THESIS_THRESHOLDS,
    high_conf_rescue: bool = True,
) -> pd.DataFrame:
    """Re-derive composite and decision from the six logged signals with the thesis policy.

    Layer 1: composite + contradiction ceiling 0.25. Layer 2: unsupported override (>0.85 BLOCK,
    >0.60 WARN if more severe). Layer 3: BLOCK -> WARN when retrieval confidence >= 0.80 and
    composite < pass threshold. The gold-similarity rescue was inactive in the run and is omitted.
    """
    x = s7(df).copy()
    clip = lambda s: s.clip(0.0, 1.0)  # noqa: E731
    comp = (
        weights["Signal_NLI"] * clip(x["Signal_NLI"])
        + weights["Signal_Numeric"] * clip(x["Signal_Numeric"])
        + weights["Signal_Entity"] * clip(x["Signal_Entity"])
        + weights["Signal_Retrieval"] * clip(x["Signal_Retrieval"])
        + weights["not_contradiction"] * clip(1.0 - x["Contradiction"])
        + weights["not_unsupported"] * clip(1.0 - x["Unsupported"])
    ).clip(0.0, 1.0)
    tp, tw = thresholds["pass"], thresholds["warn"]
    sev = {"PASS": 0, "WARN": 1, "BLOCK": 2}
    decisions = []
    for c, contra, unsup, rconf in zip(
        comp, x["Contradiction"], x["Unsupported"], x["Signal_Retrieval"], strict=True
    ):
        d = "PASS" if (c >= tp and contra <= 0.25) else "WARN" if c >= tw else "BLOCK"
        forced = "BLOCK" if unsup > 0.85 else "WARN" if unsup > 0.60 else None
        if forced and sev[forced] > sev[d]:
            d = forced
        if high_conf_rescue and d == "BLOCK" and c < tp and rconf >= 0.80:
            d = "WARN"
        decisions.append(d)
    x["composite_recomputed"] = comp
    x["decision_recomputed"] = decisions
    x["decision_matches"] = x["decision_recomputed"] == x["Gate"]
    return x


def report(df: pd.DataFrame) -> str:
    lines = ["# Thesis master run — recomputed", ""]
    lines.append(ladder_table(df).round(3).to_markdown())
    lines += ["", "## Gate (S7)", ""]
    for k, v in gate_summary(df).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## False-PASS proxies (S7)", ""]
    for k, v in false_pass_proxies(df).items():
        lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    rc = recompute_gate(df)
    comp_gap = (rc["composite_recomputed"] - rc["Gate_Composite"]).abs().max()
    lines += [
        "",
        "## Gate policy re-implementation check",
        "",
        f"- composite max |Δ|: {comp_gap:.6f}",
        f"- decision agreement: {rc['decision_matches'].mean():.4f}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=MASTER_DIR)
    ap.add_argument("--report", action="store_true", help="print the markdown report")
    args = ap.parse_args(argv)
    df = load_master(args.dir)
    if args.report:
        print(report(df))
    else:
        print(f"{len(df)} rows from {df['source_file'].nunique()} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
