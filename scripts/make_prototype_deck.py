"""Generate the synthetic 30-slide prototype lecture deck and its 60-question evaluation set.

    python -m scripts.make_prototype_deck --out data/prototype

Writes ``potassium_disorders.pptx`` (text slides, tables, two image-only slides with rendered
text, one diagram), ``questions.jsonl`` (factual, explanatory, multi-slide, numeric, follow-up and
unanswerable questions with gold slides) and ``lecture.json`` (upload metadata). The content is
invented teaching material for pipeline evaluation only — it is not medical guidance. Real client
slides plug into the same harness without code changes (``evaluation/runners/lecture_qa.py``).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.datasets.lecture_qa import LectureQuestion, write_questions  # noqa: E402

COURSE, LECTURE, TITLE, VERSION = "IM-2026", "L03", "Potassium disorders", "2026a"

# (title, bullets) — 28 text slides; slides 29–30 are image-only (added separately)
SLIDES: list[tuple[str, list[str]]] = [
    ("Potassium homeostasis", ["Normal serum potassium is 3.5 to 5.0 mmol/L.",
                               "98 percent of total body potassium is intracellular."]),
    ("Renal handling", ["The distal nephron secretes potassium under aldosterone control.",
                        "Potassium excretion rises with distal sodium delivery."]),
    ("Definition of hyperkalemia", ["Hyperkalemia is a serum potassium above 5.0 mmol/L.",
                                    "Severe hyperkalemia is above 6.5 mmol/L."]),
    ("Causes of hyperkalemia", ["Acute kidney injury and chronic kidney disease.",
                                "ACE inhibitors, ARBs, spironolactone and NSAIDs.",
                                "Pseudohyperkalemia from hemolysed samples."]),
    ("ECG changes in hyperkalemia", ["Peaked T waves appear first.",
                                     "PR prolongation and QRS widening follow.",
                                     "A sine-wave pattern precedes asystole."]),
    ("Treatment of hyperkalemia: first step", ["Give calcium gluconate 10 percent 10 mL IV to "
                                               "stabilise the myocardium.",
                                               "Calcium does not lower potassium."]),
    ("Shifting potassium", ["Insulin 10 units IV with glucose 25 g shifts potassium into cells.",
                            "Nebulised salbutamol 10 mg adds a further shift."]),
    ("Removing potassium", ["Loop diuretics and potassium binders remove potassium.",
                            "Dialysis is indicated for refractory hyperkalemia."]),
    ("Definition of hypokalemia", ["Hypokalemia is a serum potassium below 3.5 mmol/L.",
                                   "Severe hypokalemia is below 2.5 mmol/L."]),
    ("Causes of hypokalemia", ["Vomiting, diarrhoea and diuretics.",
                               "Hyperaldosteronism and renal tubular acidosis."]),
    ("ECG changes in hypokalemia", ["Flattened T waves and prominent U waves.",
                                    "ST depression and QT prolongation."]),
    ("Treatment of hypokalemia", ["Oral potassium chloride 40 mmol per day for mild cases.",
                                  "IV potassium at no more than 10 mmol per hour peripherally."]),
    ("Magnesium and potassium", ["Hypomagnesaemia causes refractory hypokalemia.",
                                 "Replace magnesium before potassium."]),
    ("Transtubular potassium gradient", ["TTKG above 7 suggests renal potassium loss.",
                                         "TTKG below 3 suggests extrarenal loss."]),
    ("Drug interactions", ["Trimethoprim blocks ENaC and raises potassium.",
                           "Digoxin toxicity is worsened by hypokalemia."]),
    ("Periodic paralysis", ["Hypokalaemic periodic paralysis follows carbohydrate loads.",
                            "Thyrotoxic periodic paralysis affects Asian men most often."]),
    ("Renal tubular acidosis", ["Type 1 RTA causes hypokalemia.",
                                "Type 4 RTA causes hyperkalemia."]),
    ("Potassium in diabetic ketoacidosis", ["Total body potassium is low despite normal serum "
                                            "values.",
                                            "Replace potassium once serum falls below "
                                            "5.5 mmol/L."]),
    ("Monitoring", ["Recheck potassium 1 hour after treatment.",
                    "Continuous ECG monitoring for potassium above 6.5 mmol/L."]),
    ("Pseudohyperkalemia", ["Fist clenching and prolonged tourniquet time raise measured "
                            "potassium.",
                            "Repeat the sample before treating."]),
    ("Dietary sources", ["Bananas, oranges and potatoes are high in potassium.",
                         "Salt substitutes contain potassium chloride."]),
    ("Potassium binders", ["Sodium zirconium cyclosilicate acts within 1 hour.",
                           "Patiromer acts within 7 hours."]),
    ("Chronic kidney disease", ["Hyperkalemia risk rises as eGFR falls below 30.",
                                "Restrict dietary potassium to 2 g per day in stage 4."]),
    ("Heart failure", ["Spironolactone raises potassium by about 0.3 mmol/L.",
                       "Monitor potassium within one week of starting."]),
    ("Summary: hyperkalemia", ["Stabilise, shift, remove.",
                               "Calcium first, then insulin with glucose, then binders."]),
    ("Summary: hypokalemia", ["Replace magnesium first.",
                              "Oral replacement unless severe or symptomatic."]),
    ("Case 1", ["A 70-year-old on ramipril has potassium 6.8 mmol/L with peaked T waves.",
                "Give calcium gluconate, then insulin with glucose."]),
    ("Case 2", ["A 30-year-old with vomiting has potassium 2.4 mmol/L and U waves.",
                "Check magnesium, start IV potassium at 10 mmol per hour."]),
]
IMAGE_SLIDES = [("ECG: hyperkalemia", "Peaked T waves\nQRS widening\nK 7.1 mmol/L"),
                ("Treatment algorithm", "Calcium gluconate -> insulin + glucose -> binders")]


def _png(text: str) -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 240), "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(text.split("\n")):
        d.text((20, 20 + 40 * i), line, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_deck(path: Path) -> Path:
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    for title, bullets in SLIDES:
        s = prs.slides.add_slide(prs.slide_layouts[1])
        s.shapes.title.text = title
        tf = s.placeholders[1].text_frame
        tf.text = bullets[0]
        for b in bullets[1:]:
            tf.add_paragraph().text = b
    for title, text in IMAGE_SLIDES:
        s = prs.slides.add_slide(prs.slide_layouts[5])
        s.shapes.title.text = title
        s.shapes.add_picture(io.BytesIO(_png(text)), Inches(1), Inches(1.5), width=Inches(6))
    path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(path))
    return path


def gold(*slides: int) -> list[str]:
    return [f"{COURSE}/{LECTURE}/s{n}" for n in slides]


def build_questions() -> list[LectureQuestion]:
    q: list[LectureQuestion] = []

    def add(qid, question, type_, slides=(), answer="", follow_up_of=None):
        q.append(LectureQuestion(id=qid, question=question, type=type_, gold_slides=gold(*slides),
                                 gold_answer=answer, follow_up_of=follow_up_of,
                                 reviewer="synthetic"))  # fmt: skip

    add("f01", "What is the normal serum potassium range?", "factual", (1,), "3.5 to 5.0 mmol/L")
    add("f02", "Which ECG change appears first in hyperkalemia?", "factual", (5,), "Peaked T waves")
    add("f03", "What is the first step in treating hyperkalemia?", "factual", (6, 25),
        "Calcium gluconate 10 percent 10 mL IV")
    add("f04", "What defines severe hypokalemia?", "factual", (9,), "Below 2.5 mmol/L")
    add("f05", "Which electrolyte must be replaced before potassium in refractory hypokalemia?",
        "factual", (13, 26), "Magnesium")
    add("f06", "How quickly does sodium zirconium cyclosilicate act?", "factual", (22,),
        "Within 1 hour")
    add("f07", "Which type of renal tubular acidosis causes hyperkalemia?", "factual", (17,),
        "Type 4 RTA")
    add("f08", "What does a TTKG above 7 suggest?", "factual", (14,), "Renal potassium loss")
    add("f09", "Which drug blocks ENaC and raises potassium?", "factual", (15,), "Trimethoprim")
    add("f10", "What ECG findings occur in hypokalemia?", "factual", (11,),
        "Flattened T waves, prominent U waves, ST depression, QT prolongation")
    add("e01", "Why does calcium gluconate not lower the potassium level?", "explanatory", (6,),
        "It stabilises the myocardium; it does not shift or remove potassium")
    add("e02", "Why is magnesium replaced before potassium?", "explanatory", (13,),
        "Hypomagnesaemia causes refractory hypokalemia")
    add("e03", "Why can a hemolysed sample show hyperkalemia?", "explanatory", (4, 20),
        "Pseudohyperkalemia from cell lysis; repeat the sample")
    add("e04", "Why does total body potassium fall in diabetic ketoacidosis despite a normal "
        "serum level?", "explanatory", (18,), "Shift out of cells masks depletion")
    add("m01", "Describe the full treatment sequence for severe hyperkalemia.", "multi_slide",
        (6, 7, 8, 25), "Stabilise with calcium, shift with insulin/glucose and salbutamol, "
                       "remove with binders, diuretics or dialysis")
    add("m02", "Compare the ECG changes of hyperkalemia and hypokalemia.", "multi_slide", (5, 11),
        "Peaked T, PR prolongation, QRS widening vs flattened T, U waves, ST depression")
    add("m03", "How does chronic kidney disease change hyperkalemia risk and management?",
        "multi_slide", (4, 23), "Risk rises below eGFR 30; dietary restriction to 2 g per day")
    add("m04", "Which drugs in the lecture raise potassium?", "multi_slide", (4, 15, 24),
        "ACE inhibitors, ARBs, spironolactone, NSAIDs, trimethoprim")
    add("n01", "What dose of calcium gluconate is given for hyperkalemia?", "numeric", (6,),
        "10 mL of 10 percent IV")
    add("n02", "What is the maximum peripheral IV potassium rate?", "numeric", (12, 28),
        "10 mmol per hour")
    add("n03", "How much does spironolactone raise potassium on average?", "numeric", (24,),
        "About 0.3 mmol/L")
    add("n04", "What insulin and glucose doses shift potassium?", "numeric", (7,),
        "Insulin 10 units IV with glucose 25 g")
    add("n05", "Below which serum potassium is replacement started in DKA?", "numeric", (18,),
        "Below 5.5 mmol/L")
    add("n06", "What potassium level warrants continuous ECG monitoring?", "numeric", (19,),
        "Above 6.5 mmol/L")
    add("u01", "What is the recommended dose of amiodarone for atrial fibrillation?",
        "unanswerable")
    add("u02", "How is hyponatremia corrected?", "unanswerable")
    add("u03", "What did the 2024 KDIGO guideline change about potassium binders?",
        "unanswerable")
    add("u04", "Which lecture covers sodium disorders?", "unanswerable")
    add("fu01", "And what is the next step after that?", "follow_up", (7, 25),
        "Insulin 10 units IV with glucose 25 g", follow_up_of="f03")
    add("fu02", "What ECG changes does it cause?", "follow_up", (11,),
        "Flattened T waves, U waves", follow_up_of="f04")
    add("fu03", "Why is that?", "follow_up", (13,), "Hypomagnesaemia causes refractory "
        "hypokalemia", follow_up_of="f05")
    add("fu04", "How fast does the other binder act?", "follow_up", (22,), "Patiromer within "
        "7 hours", follow_up_of="f06")
    add("i01", "What potassium value is shown on the ECG slide?", "factual", (29,),
        "7.1 mmol/L")
    add("i02", "What is the order of the treatment algorithm slide?", "factual", (30,),
        "Calcium gluconate, insulin with glucose, binders")
    return q


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "prototype")
    a = ap.parse_args(argv)
    deck = build_deck(a.out / "potassium_disorders.pptx")
    n = write_questions(build_questions(), a.out / "questions.jsonl")
    (a.out / "lecture.json").write_text(json.dumps({
        "course_id": COURSE, "lecture_id": LECTURE, "lecture_title": TITLE,
        "academic_year": "2026", "version": VERSION}, indent=1))
    print(json.dumps({"deck": str(deck), "slides": len(SLIDES) + len(IMAGE_SLIDES),
                      "questions": n}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
