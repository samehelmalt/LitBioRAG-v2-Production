"""The prototype harness: synthetic deck + question set → run → report, with test doubles."""

import json

import pytest

pytest.importorskip("pptx")
pytest.importorskip("tantivy")
pytest.importorskip("faiss")

from app.core.budget import UsageLedger  # noqa: E402
from app.service.pipeline import AnswerPipeline  # noqa: E402
from evaluation.datasets.lecture_qa import (  # noqa: E402
    LectureQuestion,
    load_questions,
    slide_key,
    summarize_set,
)
from scripts import make_prototype_deck, run_lecture_qa  # noqa: E402


def test_question_set_schema_and_validation(tmp_path):
    with pytest.raises(ValueError):
        LectureQuestion(id="x", question="q", type="factual")  # answerable without gold
    with pytest.raises(ValueError):
        LectureQuestion(id="x", question="q", type="follow_up", gold_slides=["a/b/s1"])
    u = LectureQuestion(id="u", question="q", type="unanswerable")
    assert u.answerable is False
    assert slide_key("IM-2026/L03/s4/v2026a/abcd") == "IM-2026/L03/s4"
    qs = make_prototype_deck.build_questions()
    s = summarize_set(qs)
    assert s["n"] >= 30 and s["by_type"]["follow_up"] >= 3 and s["by_type"]["unanswerable"] >= 3
    assert all(q.type in ("factual", "explanatory", "multi_slide", "numeric", "follow_up",
                          "unanswerable") for q in qs)


def test_prototype_deck_and_full_run(fixture_bundle, tmp_path):
    out = tmp_path / "proto"
    assert make_prototype_deck.main(["--out", str(out)]) == 0
    deck = out / "potassium_disorders.pptx"
    questions = load_questions(out / "questions.jsonl")
    lecture = json.loads((out / "lecture.json").read_text())
    pl = AnswerPipeline(fixture_bundle, profile="test", ledger=UsageLedger(None),
                        kb_root=tmp_path / "kb", scope_profile="education")  # fmt: skip
    summary = run_lecture_qa.evaluate(pl, deck, lecture, out / "questions.jsonl",
                                      tmp_path / "run")  # fmt: skip
    assert summary["n"] == len(questions)
    assert (tmp_path / "run" / "rows.jsonl").exists() and (tmp_path / "run" / "report.md").exists()
    # with the lexical test doubles the pipeline must still: retrieve gold slides for direct
    # factual questions, keep every delivered citation valid, and abstain on unanswerables
    assert summary["retrieval"]["recall_at_5"] >= 0.6
    assert summary["citations"]["validity"] == 1.0 and summary["citations"]["errors"] == 0
    # the lexical test double answers one unanswerable question on word overlap ("sodium");
    # the target (0 false answers) is enforced on real-model runs, not on the doubles
    assert summary["answers"]["unanswerable_abstained"] >= 0.5
    assert summary["versions"]["question_set"]["n"] == len(questions)
    rows = [json.loads(line) for line in (tmp_path / "run" / "rows.jsonl").read_text().splitlines()]
    fu = next(r for r in rows if r["id"] == "fu01")
    assert fu["type"] == "follow_up" and fu["human_label"] == ""
    img = next(r for r in rows if r["id"] == "i01")
    assert img["gold_slides"] == ["IM-2026/L03/s29"]  # image slide: needs OCR to be answered
    assert all(c.startswith("IM-2026/L03/s") for r in rows for c in r["cited_slides"])
