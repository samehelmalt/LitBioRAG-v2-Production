from app.verification.entities import RegexBiomedEntityExtractor, entity_consistency
from app.verification.numeric import extract_numbers, numeric_consistency

EV = (
    "MAP3K3 (MEKK3) is required for NF-kappaB signalling. Knockdown of MAP3K3 reduced p65 "
    "nuclear translocation by 61% (p < 0.01) in 24 patients treated with 5 mg atorvastatin."
)
C_OK = "MAP3K3 is required for NF-kappaB signalling."
C_SWAP = "MAP3K7 is required for NF-kappaB signalling."


def test_entity_extraction_and_consistency():
    x = RegexBiomedEntityExtractor()
    ents = x.extract(EV)
    assert {"map3k3", "mekk3", "nfkappab", "atorvastatin"} <= ents
    assert "the" not in ents and "dna" not in ents and "nf" not in ents
    assert entity_consistency(C_OK, [EV], x) == (2, 2, ())
    checked, ok, missing = entity_consistency(C_SWAP, [EV], x)
    assert checked == 2 and ok == 1 and missing == ("map3k7",)
    assert entity_consistency("The outcome was measured.", [EV], x) == (0, 0, ())


def test_number_extraction_classes():
    nums = {n.raw: n for n in extract_numbers(EV)}
    classes = {n.unit_class for n in nums.values()}
    assert {"percent", "p_value", "count", "mass"} <= classes
    pv = next(n for n in nums.values() if n.unit_class == "p_value")
    assert pv.value == 0.01
    assert any(n.value == 61 and n.unit_class == "percent" for n in nums.values())
    ci = extract_numbers("HR 0.72 (95% CI 0.61-0.85)")
    assert any(n.unit_class == "ratio" and n.value == 0.72 for n in ci)
    assert not any(n.value == 95 for n in ci)  # the 95 of '95% CI' is not a finding


def test_numeric_consistency_catches_altered_and_misclassed_numbers():
    assert numeric_consistency("Translocation fell by 61%.", [EV]) == (1, 1, ())
    checked, ok, mism = numeric_consistency("Translocation fell by 16%.", [EV])
    assert (checked, ok) == (1, 0) and mism == ("16%",)
    # 61 patients is not the same finding as 61% — unit class must match
    checked, ok, mism = numeric_consistency("61 patients were treated.", [EV])
    assert ok == 0
    # tolerance: 60.5% matches 61% within 2%
    assert numeric_consistency("Translocation fell by 60.5%.", [EV])[1] == 1
    assert numeric_consistency("No numbers here.", [EV]) == (0, 0, ())
