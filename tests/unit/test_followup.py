"""Follow-up resolution uses only the last delivered turn; withheld turns never resolve anything."""

from app.service.followup import is_follow_up, resolve
from app.verification.entities import RegexBiomedEntityExtractor

X = RegexBiomedEntityExtractor()
H = ["Q: What is the first step in treating hyperkalemia? → delivered answer: Give calcium "
     "gluconate 10 percent 10 mL IV to stabilise the myocardium."]


def test_detects_follow_ups():
    assert is_follow_up("Why is that?") and is_follow_up("And what is the next step after that?")
    assert is_follow_up("How fast does the other binder act?")
    assert not is_follow_up("What is the normal serum potassium range?")


def test_rewrites_with_entities_from_the_delivered_turn():
    r = resolve("Why is that?", H, X.extract)
    assert r.interpreted and "hyperkalemia" in r.question and "gluconate" in r.question.lower()
    assert r.based_on.startswith("What is the first step") and "not evidence" in r.note
    assert resolve("What is the normal serum potassium range?", H, X.extract).interpreted is False


def test_withheld_turns_do_not_resolve():
    r = resolve("Why is that?", ["Q: My mother takes warfarin; what dose? → (no answer was "
                                 "delivered)"], X.extract)
    assert r.interpreted is False and "no delivered answer" in r.note
    assert resolve("Why is that?", [], X.extract).interpreted is False
