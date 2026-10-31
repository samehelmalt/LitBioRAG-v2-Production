"""Confusable-entity rules cover genes, drugs, conditions and numbered families alike."""

import pytest

from app.core.symbols import neighbour_reason, normalize
from app.verification.entities import RegexBiomedEntityExtractor

NEIGHBOURS = [
    ("MAP3K3", "MAP3K7", "numbered_family"),
    ("MAP3K3", "MAPK3", "numbered_family"),
    ("IL-6", "IL-17", "numbered_family"),
    ("CD4", "CD8", "numbered_family"),
    ("HER2", "HER3", "numbered_family"),
    ("BRCA1", "BRCA2", "numbered_family"),
    ("COX-1", "COX-2", "numbered_family"),
    ("trisomy 21", "trisomy 18", "numbered_family"),
    ("type 1 diabetes", "type 2 diabetes", "numbered_family"),
    ("Type II diabetes", "type 1 diabetes mellitus", "numbered_family"),
    ("hepatitis B", "hepatitis C", "letter_variant"),
    ("hypokalemia", "hyperkalemia", "antonym_prefix"),
    ("hyperthyroidism", "hypothyroidism", "antonym_prefix"),
    ("bradycardia", "tachycardia", "antonym_prefix"),
    ("atorvastatin", "simvastatin", "drug_class:statin"),
    ("lisinopril", "enalapril", "drug_class:pril"),
    ("losartan", "valsartan", "drug_class:sartan"),
    ("hydroxyzine", "hydralazine", "confusable_group"),
    ("dopamine", "dobutamine", "confusable_group"),
    ("Crohn's disease", "celiac disease", "confusable_group"),
    ("prednisone", "prednisolone", "confusable_group"),
]
NOT_NEIGHBOURS = [
    ("heart attack", "myocardial infarction"),  # synonyms
    ("TAK1", "MAP3K7"),  # alias
    ("T2DM", "type 2 diabetes"),  # abbreviation
    ("MEKK3", "MAP3K3"),
    ("COVID-19", "COVID"),  # same name with/without number
    ("hepatitis", "hepatitis B"),  # generic vs specific
    ("diabetes", "type 2 diabetes"),
    ("aspirin", "warfarin"),  # different drugs, different classes
    ("MAP3K3", "MAP3K3"),
]


@pytest.mark.parametrize("a,b,reason", NEIGHBOURS)
def test_neighbours(a, b, reason):
    assert neighbour_reason(a, b) == reason
    assert neighbour_reason(b, a) == reason


@pytest.mark.parametrize("a,b", NOT_NEIGHBOURS)
def test_not_neighbours(a, b):
    assert neighbour_reason(a, b) is None


def test_normalize_roman_and_synonyms():
    assert normalize("Type II Diabetes Mellitus") == "type2diabetes"
    assert normalize("T2DM") == "type2diabetes"
    assert normalize("heart attack") == normalize("MI") == "myocardialinfarction"
    assert normalize("HBV") == normalize("hepatitis B")


def test_extractor_mentions_cover_all_kinds():
    x = RegexBiomedEntityExtractor()
    text = ("Does atorvastatin reduce LDL in type 2 diabetes mellitus patients with hypokalemia, "
            "hepatitis B and chronic kidney disease? IL-6 and CD4 counts; stage III colon cancer; "
            "MAP3K3 activates NF-kappaB.")
    ms = {m.norm: m.kind for m in x.mentions(text)}
    assert ms["atorvastatin"] == "drug"
    assert ms["type2diabetes"] == "condition"
    assert ms["hypokalemia"] == "condition"
    assert ms["hepatitisb"] == "condition"
    assert ms["chronickidneydisease"] == "condition"
    assert ms["il6"] == "gene" and ms["cd4"] == "gene" and ms["map3k3"] == "gene"
    assert ms["coloncancer"] == "condition"
    assert ms["nfkappab"] == "pathway"
    assert "hepatitis" not in ms  # nested mention dropped in favour of 'hepatitis B'


def test_extractor_ignores_common_words():
    x = RegexBiomedEntityExtractor()
    ents = x.extract("The diagnosis and prognosis were evaluated with attention to the outcome.")
    assert ents == set()


def test_rules_are_config_driven(tmp_path, monkeypatch):
    from app.core import symbols

    cfg = tmp_path / "entities.yaml"
    cfg.write_text("confusable_groups:\n  - [foo, bar]\nsynonyms: {baz: foo}\n")
    monkeypatch.setenv("LITBIORAG_ENTITIES_CONFIG", str(cfg))
    symbols._rules_for.cache_clear()
    try:
        assert neighbour_reason("foo", "bar") == "confusable_group"
        assert neighbour_reason("baz", "bar") == "confusable_group"  # via synonym
        assert neighbour_reason("atorvastatin", "simvastatin") is None  # no suffixes in this file
    finally:
        monkeypatch.delenv("LITBIORAG_ENTITIES_CONFIG")
        symbols._rules_for.cache_clear()
