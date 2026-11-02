"""Prototype evaluation: upload a deck into a fresh knowledge base, run the question set, report.

    python -m scripts.run_lecture_qa --bundle data/artifacts/<id> --profile local-8b \
        --deck data/prototype/potassium_disorders.pptx --lecture data/prototype/lecture.json \
        --questions data/prototype/questions.jsonl \
        --out evaluation/benchmarks/runs/<name>/lecture_qa

Use ``python -m scripts.make_prototype_deck`` to create the synthetic 30-slide prototype. Real
client slides use the same command with their own deck and reviewed question set.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.budget import UsageLedger  # noqa: E402
from app.service.pipeline import AnswerPipeline  # noqa: E402
from evaluation.datasets.lecture_qa import load_questions, slide_key, summarize_set  # noqa: E402
from evaluation.runners.lecture_qa import run_lecture_qa, write_run  # noqa: E402


def make_slide_lookup(indexer):
    """citation id → (slide key, slide text) from the KB store."""
    by_cit: dict[str, tuple[str, str]] = {}
    for did in indexer.store.doc_ids_where():
        d = indexer.store.get_document(did)
        if d is None or d.file is None:
            continue
        f = d.file
        cid = (f"SLIDE:{f.lecture_id}:{f.slide_number}" if f.lecture_id
               else f"FILE:{f.file_id}:{f.locator}")
        by_cit[cid] = (slide_key(f.canonical_id), d.text)
    return lambda cid: by_cit.get(cid)


def evaluate(pipeline: AnswerPipeline, deck: Path, lecture: dict, questions_path: Path,
             out: Path, *, kb_name: str = "prototype-eval") -> dict:  # fmt: skip
    ix = pipeline.kb
    if ix is None:
        raise RuntimeError("pipeline has no knowledge-base root (kb_root)")
    kb = ix.registry.create(name=kb_name, owner="evaluator")
    ix.add_file(kb["kb_id"], deck, file_name=deck.name, uploaded_by="evaluator", meta=lecture)
    questions = load_questions(questions_path)
    rows = run_lecture_qa(pipeline, questions, kb_ids=[kb["kb_id"]],
                          slide_lookup=make_slide_lookup(ix))  # fmt: skip
    summary = write_run(rows, out, versions={**pipeline.versions.as_dict(),
                                             "question_set": summarize_set(questions),
                                             "deck": deck.name, "lecture": lecture})
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--deck", type=Path, required=True)
    ap.add_argument("--lecture", type=Path, required=True, help="JSON with course/lecture fields")
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kb-root", type=Path, default=None, help="default: a temporary directory")
    a = ap.parse_args(argv)
    kb_root = a.kb_root or Path(tempfile.mkdtemp(prefix="lecture-qa-"))
    pl = AnswerPipeline(a.bundle, profile=a.profile, ledger=UsageLedger(None), kb_root=kb_root,
                        scope_profile="education")  # fmt: skip
    s = evaluate(pl, a.deck, json.loads(a.lecture.read_text()), a.questions, a.out)
    print(json.dumps({k: s[k] for k in ("n", "retrieval", "citations", "answers")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
