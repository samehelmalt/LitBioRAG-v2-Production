import re
from pathlib import Path

import pytest

from app.core.verdicts import ClaimVerdict, ConflictRecord, VerificationReport
from app.generation.contract import Claim, GeneratedAnswer
from app.safety.gate import Action, GateConfig, decide
from app.safety.repair import repair_answer
from app.safety.scope import classify_scope

Q = "What does MAP3K3 do in NF-kappaB signalling?"


def ans(*claims, abstained=False, reason=None):
    cs = [
        Claim(id=f"C{i}", text=t, citations=c, critical=k)
        for i, (t, c, k) in enumerate(claims, 1)
    ]
    return GeneratedAnswer(answer=" ".join(t for t, _, _ in claims), claims=cs, abstained=abstained,
                           abstain_reason=reason, question=Q)  # fmt: skip


def verdict(i, text, label="supported", critical=True, cited=("PMID:1",), supporting=("PMID:1",),
            **kw):  # fmt: skip
    return ClaimVerdict(claim_id=f"C{i}", text=text, label=label, critical=critical,
                        cited_ids=tuple(cited), resolved_ids=tuple(cited),
                        supporting_ids=tuple(supporting), support=0.9, **kw)  # fmt: skip


def report(*vs, conflicts=(), grounding=1.0, fabricated=()):
    r = VerificationReport.from_verdicts(list(vs), list(conflicts), verifier="t")
    r.query_entity_grounding = grounding
    if fabricated:
        r.fabricated_citations = tuple(fabricated)
    return r


GOOD = ans(("MAP3K3 is required for NF-kappaB signalling.", ["PMID:1"], True))


def test_rule_order_each_rule_reachable():
    cfg = GateConfig()
    # R1 scope
    d = decide(report(verdict(1, "x")), GOOD, question="What dose should I take for my mother?")
    assert d.action == Action.ESCALATE and d.rule == "R1-scope" and d.internal_state == "BLOCK"
    # R2 generator abstained
    d = decide(report(), ans(abstained=True, reason="insufficient_evidence"), config=cfg)
    assert d.action == Action.ABSTAIN and d.rule == "R2-generator-abstained"
    # R3 fabricated
    d = decide(report(verdict(1, "x"), fabricated=["PMID:9"]), GOOD, config=cfg)
    assert d.rule == "R3-fabricated-citation" and d.action == Action.ABSTAIN
    # R4 critical uncited / unsupported citation
    d = decide(report(verdict(1, "x", cited=(), supporting=())), GOOD, config=cfg)
    assert d.rule == "R4-critical-uncited"
    d = decide(report(verdict(1, "x", supporting=())), GOOD, config=cfg)
    assert d.rule == "R4-critical-uncited"
    # R5 contradicted, with and without conflict record
    d = decide(report(verdict(1, "x", label="contradicted")), GOOD, config=cfg)
    assert d.rule == "R5-critical-contradicted" and "contradicts" in d.reason
    d = decide(report(verdict(1, "x", label="contradicted"),
                      conflicts=[ConflictRecord(claim_id="C1")]), GOOD, config=cfg)  # fmt: skip
    assert "conflicting" in d.reason and "conflicting" in d.answer_text.lower()
    # R6 substitution / entity mismatch
    d = decide(report(verdict(1, "x", near_neighbour_substitution=True)), GOOD, config=cfg)
    assert d.rule == "R6-entity" and "substitution" in d.reason
    d = decide(report(verdict(1, "x", entity_checked=1, entity_ok=0)), GOOD, config=cfg)
    assert d.rule == "R6-entity" and "mismatch" in d.reason
    # R7 numeric
    d = decide(report(verdict(1, "x", numeric_checked=1, numeric_ok=0)), GOOD, config=cfg)
    assert d.rule == "R7-numeric"
    # R8 grounding → CLARIFY
    d = decide(report(verdict(1, "x"), grounding=0.0), GOOD, config=cfg)
    assert d.action == Action.CLARIFY and d.rule == "R8-grounding"
    # R9 sufficiency
    d = decide(report(verdict(1, "x")), GOOD, sufficiency=0.1, config=cfg)
    assert d.rule == "R9-sufficiency" and d.action == Action.ABSTAIN
    # R10 too many unsupported non-critical claims
    many = ans(("a", ["PMID:1"], True), ("b", [], False), ("c", [], False))
    d = decide(report(verdict(1, "a"), verdict(2, "b", "unsupported", False, (), ()),
                      verdict(3, "c", "unsupported", False, (), ())), many, config=cfg)  # fmt: skip
    assert d.rule == "R10-too-many-unsupported"
    # R10 repair (1 of 3 claims removable, under the 0.34 cap) → caution, removal listed,
    # citations kept
    three = ans(("MAP3K3 is required.", ["PMID:1"], True), ("Something extra.", [], False),
                ("It acts upstream.", ["PMID:1"], True))  # fmt: skip
    rep3 = report(verdict(1, "MAP3K3 is required."),
                  verdict(2, "Something extra.", "unsupported", False, (), ()),
                  verdict(3, "It acts upstream."))  # fmt: skip
    d = decide(rep3, three, sufficiency=0.9, config=cfg)
    assert d.action == Action.ANSWER_WITH_CAUTION and d.rule == "R10-repair"
    assert d.removed_claims == ["Something extra."] and "[PMID:1]" in d.answer_text
    assert "Something extra" not in d.answer_text
    # R11 conflict on a supported claim → caution
    d = decide(report(verdict(1, "x"), conflicts=[ConflictRecord(claim_id="C1")]), GOOD,
               sufficiency=0.9, config=cfg)  # fmt: skip
    assert d.rule == "R11-conflict" and d.action == Action.ANSWER_WITH_CAUTION
    # R12 confidence
    d = decide(report(verdict(1, "x")), GOOD, sufficiency=0.9, config=cfg)
    assert d.rule == "R12-confidence" and d.action == Action.ANSWER and d.internal_state == "PASS"
    assert d.confidence is not None and d.trace[-1].startswith("R12")
    d = decide(report(verdict(1, "x")), GOOD, sufficiency=0.4, config=cfg)
    assert d.action == Action.ANSWER_WITH_CAUTION  # weakest check bounds confidence


def test_fitted_confidence_model_is_used():
    cfg = GateConfig(confidence_model={"weights": {"entailment_rate": 8.0}, "bias": -4.0},
                     confidence_answer=0.9)  # fmt: skip
    d = decide(report(verdict(1, "x")), GOOD, sufficiency=0.9, config=cfg)
    assert d.confidence == pytest.approx(1 / (1 + 2.718281828 ** -4.0), rel=1e-3)
    assert d.action == Action.ANSWER


def test_gate_yaml_loads_and_is_marked_uncalibrated():
    cfg = GateConfig.from_yaml("configs/gate.yaml")
    assert cfg.calibrated_on == "" and cfg.confidence_model is None
    assert cfg.costs["false_pass"] > cfg.costs["false_block"]


def test_scope_classifier():
    assert classify_scope("Does MAP3K3 regulate NF-kappaB?").in_scope
    mother = classify_scope("My mother has chest pain, what should she take?")
    assert mother.kind == "patient_specific"
    assert classify_scope("What dose should I take of metformin?").kind == "personal_dosing"
    assert classify_scope("Someone is not breathing right now").kind == "emergency"
    assert "clinician" in classify_scope("Should I stop my statin?").message


def test_repair_keeps_order_and_citations():
    a = ans(("First.", ["PMID:1"], True), ("Second.", [], False), ("Third.", ["PMID:2"], True))
    text, removed = repair_answer(a, ["C2"])
    assert text == "First. [PMID:1] Third. [PMID:2]" and removed == ["Second."]


def test_no_gold_labels_in_runtime_code():
    """app/ must never read ground truth; evaluation/ is the only place labels exist."""
    pattern = re.compile(r"\b(yn_label|yn_gold|gold_pmids?|ground_truth|proxy_acceptable)\b")
    offenders = [
        str(p) for p in Path("app").rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []
    assert not any("from evaluation" in p.read_text() or "import evaluation" in p.read_text()
                   for p in Path("app").rglob("*.py"))  # fmt: skip
