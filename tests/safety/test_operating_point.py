"""Target-precision operating point: release automatically only where the calibration rows show
the requested precision; otherwise everything goes to caution/review."""

from scripts.calibrate_gate import choose_threshold_for_precision


def test_threshold_reaches_target_with_max_coverage():
    conf = [0.99, 0.98, 0.97, 0.96, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4]
    ok = [True, True, True, True, True, True, True, True, True, False, False, False]
    t, point = choose_threshold_for_precision(conf, ok, 0.95, min_released=3)
    assert t == 0.7 and point["precision"] == 1.0 and point["coverage"] == 9 / 12
    t, point = choose_threshold_for_precision(conf, ok, 0.90, min_released=3)
    assert t == 0.6 and abs(point["precision"] - 0.9) < 1e-9


def test_unreachable_target_releases_nothing_automatically():
    conf = [0.9, 0.8, 0.7, 0.6]
    ok = [False, True, False, True]
    t, point = choose_threshold_for_precision(conf, ok, 0.95, min_released=2)
    assert t == 1.01 and point["precision"] is None and point["released"] == 0
