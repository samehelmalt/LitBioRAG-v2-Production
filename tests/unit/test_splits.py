import pytest

from evaluation.datasets.loaders import QueryRecord
from evaluation.splits import SplitError, SplitManifest, build_manifest, verify


def rec(i, pmids, qtype="yesno", ds="BioASQ", text=None):
    return QueryRecord(
        id=f"r{i}",
        question=text or f"question number {i}?",
        gold_pmids=tuple(pmids),
        dataset=ds,
        question_type=qtype,
    )


def test_pmid_groups_stay_together_and_pins_hold():
    recs = [rec(0, ["1"]), rec(1, ["1", "2"]), rec(2, ["2"]), rec(3, ["9"]), rec(4, ["10"])]
    m = build_manifest(recs, pinned={"r0": "test"}, seed=1)
    assert m.assignments["r0"] == m.assignments["r1"] == m.assignments["r2"] == "test"
    verify(m, recs)


def test_deterministic_for_seed():
    recs = [rec(i, [str(100 + i)]) for i in range(40)]
    a = build_manifest(recs, seed=7).assignments
    b = build_manifest(recs, seed=7).assignments
    c = build_manifest(recs, seed=8).assignments
    assert a == b
    assert a != c


def test_verify_rejects_text_overlap_across_splits():
    recs = [rec(0, ["1"], text="same text?"), rec(1, ["2"], text="Same text")]
    m = SplitManifest("v", 0, {"r0": "test", "r1": "tuning"})
    with pytest.raises(SplitError):
        verify(m, recs)


def test_verify_rejects_pmid_overlap_across_splits():
    recs = [rec(0, ["1"]), rec(1, ["1"])]
    m = SplitManifest("v", 0, {"r0": "test", "r1": "tuning"})
    with pytest.raises(SplitError):
        verify(m, recs)


def test_conflicting_pins_raise():
    recs = [rec(0, ["1"]), rec(1, ["1"])]
    with pytest.raises(SplitError):
        build_manifest(recs, pinned={"r0": "test", "r1": "tuning"})


def test_manifest_roundtrip(tmp_path):
    recs = [rec(i, [str(i)]) for i in range(10)]
    m = build_manifest(recs, seed=3, pinned={"r0": "test"})
    p = tmp_path / "m.json"
    m.save(p)
    m2 = SplitManifest.load(p)
    assert m2.assignments == m.assignments and m2.pinned == m.pinned and m2.seed == 3
