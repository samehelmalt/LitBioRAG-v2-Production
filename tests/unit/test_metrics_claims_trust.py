import pytest

from evaluation.metrics.claims import ClaimVerdict, summarize
from evaluation.metrics.performance import latency_summary, percentile
from evaluation.metrics.trust import (
    brier,
    coverage_risk_curve,
    ece,
    gate_confusion,
    risk_at_coverage,
)


def test_claim_summary():
    vs = [
        ClaimVerdict("supported", ("1",), ("1",), ("1",), numeric_checked=2, numeric_ok=2),
        ClaimVerdict("unsupported", ("9",), (), ()),
        ClaimVerdict("contradicted", (), (), (), critical=True, entity_checked=1, entity_ok=0),
        ClaimVerdict("supported", ("1", "2"), ("1", "2"), ("1",)),
    ]
    s = summarize(vs)
    assert s.n_claims == 4
    assert s.entailment_rate == 0.5 and s.unsupported_rate == 0.25 and s.contradiction_rate == 0.25
    assert s.critical_contradicted == 1
    assert s.citation_resolution_rate == pytest.approx(3 / 4)  # cited 4, resolved 3
    assert s.citation_support_rate == pytest.approx(2 / 4)
    assert s.uncited_claim_rate == 0.25
    assert s.numeric_fidelity == 1.0 and s.entity_fidelity == 0.0
    assert summarize([]).n_claims == 0 and summarize([]).numeric_fidelity is None


def test_claim_label_validation():
    with pytest.raises(ValueError):
        ClaimVerdict("maybe")


def test_gate_confusion_hand_computed():
    d = ["PASS", "PASS", "WARN", "BLOCK", "BLOCK", "BLOCK"]
    ok = [True, False, False, False, True, False]
    g = gate_confusion(d, ok)
    assert g.false_pass_rate == 0.5
    assert g.false_warn_rate == 1.0
    assert g.false_block_rate == pytest.approx(1 / 3)
    assert g.abstention_precision == pytest.approx(2 / 3)
    assert g.abstention_recall == pytest.approx(2 / 4)
    assert g.unacceptable_released_rate == pytest.approx(2 / 4)
    assert g.coverage == 0.5 and g.block_rate == 0.5


def test_calibration_metrics():
    conf = [0.9, 0.8, 0.2, 0.1]
    ok = [True, True, False, False]
    assert brier(conf, ok) == pytest.approx((0.01 + 0.04 + 0.04 + 0.01) / 4)
    # bins of width 0.1: [0.9]->acc 1 conf .9; [0.8]->1,.8; [0.2]->0,.2; [0.1]->0,.1
    assert ece(conf, ok, bins=10) == pytest.approx((0.1 + 0.2 + 0.2 + 0.1) / 4)
    assert ece([1.0, 0.0], [True, False]) == 0.0


def test_coverage_risk_curve():
    conf = [0.9, 0.7, 0.7, 0.3]
    ok = [True, False, True, False]
    curve = coverage_risk_curve(conf, ok)
    assert curve == [(0.9, 0.25, 0.0), (0.7, 0.75, pytest.approx(1 / 3)), (0.3, 1.0, 0.5)]
    assert risk_at_coverage(curve, 0.5) == pytest.approx(1 / 3)
    assert risk_at_coverage(curve, 1.1) is None


def test_percentiles():
    xs = [1, 2, 3, 4, 5]
    assert percentile(xs, 50) == 3 and percentile(xs, 0) == 1 and percentile(xs, 100) == 5
    s = latency_summary([10.0, 20.0, 30.0, 100.0])
    assert s.p50 == 25.0 and s.max == 100.0 and s.mean == 40.0
