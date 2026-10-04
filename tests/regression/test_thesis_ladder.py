"""The harness must reproduce the thesis numbers recorded in docs/THESIS_BASELINE.md §10."""

import pytest

pytest.importorskip("pandas")

from evaluation.thesis_import import (  # noqa: E402
    BLOCK_POST_GATE_CONSTANT,
    false_pass_proxies,
    gate_summary,
    ladder_table,
    load_master,
    recompute_gate,
)


@pytest.fixture(scope="module")
def df():
    return load_master()


def test_row_and_file_counts(df):
    assert len(df) == 4752
    assert df["source_file"].nunique() == 18
    assert df["Query"].nunique() == 24


def test_ladder_recall_matches_thesis(df):
    t = ladder_table(df)
    expected = {
        "S-1": 0.000, "S0": 0.472, "S1": 0.604, "S2": 0.618, "S3": 0.583, "S3a": 0.660,
        "S4": 0.583, "S5v1": 0.583, "S5": 0.528, "S6": 0.583, "S7": 0.583,
    }  # fmt: skip
    for step, val in expected.items():
        assert round(float(t.loc[step, "recall_k_norm"]), 3) == val, step
    assert round(float(t.loc["S3a", "mrr_norm"]), 3) == 0.566
    assert round(float(t.loc["S6", "nli_faithfulness"]), 3) == 0.138


def test_cross_encoder_convergence(df):
    s3a = df[df["Step"] == "S3a"].groupby("Retrieval_Model")["Recall_K_Norm"].mean()
    assert set(s3a.round(4).unique()) == {0.6597}


def test_gate_distribution_and_effect(df):
    g = gate_summary(df)
    assert g["n"] == 432
    assert g["counts"] == {"BLOCK": 221, "WARN": 147, "PASS": 64}
    assert round(g["delta_mean"], 3) == 0.452
    assert g["block_post_gate_values"] == [BLOCK_POST_GATE_CONSTANT]


def test_false_pass_proxies(df):
    f = false_pass_proxies(df)
    assert f["pass_rows"] == 64
    assert (f["pass_yesno_rows"], f["pass_yesno_wrong"]) == (25, 14)
    assert f["pass_gold_not_retrieved"] == 22
    assert 0 < f["proxy_false_pass_rate"] <= 1


def test_gate_policy_reimplementation_matches_logged_decisions(df):
    rc = recompute_gate(df)
    assert (rc["composite_recomputed"] - rc["Gate_Composite"]).abs().max() < 1e-3
    assert rc["decision_matches"].mean() >= 0.98
