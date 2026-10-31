"""Course slides as a content source: extraction, diff, course filter, slide citations."""

import json

import pytest

pptx = pytest.importorskip("pptx")

from pptx.util import Inches  # noqa: E402

from app.core.documents import SourceType  # noqa: E402
from app.evidence.selector import select_evidence  # noqa: E402
from app.generation.prompts import render_evidence  # noqa: E402
from app.reranking.base import LexicalOverlapReranker  # noqa: E402
from app.retrieval.bm25 import BM25Index  # noqa: E402
from app.retrieval.dense import DenseIndex  # noqa: E402
from app.retrieval.encoders import HashEncoder  # noqa: E402
from app.retrieval.pipeline import HybridRetriever, RetrievalConfig  # noqa: E402
from app.retrieval.store import MetadataStore  # noqa: E402
from scripts import build_indexes  # noqa: E402
from scripts.corpus import pptx_decks  # noqa: E402
from tests.retrieval.fixture_corpus import fixture_documents  # noqa: E402

SLIDES = [
    ("Atrial Fibrillation", None),  # 1: section header (title only)
    ("Definition", "Atrial fibrillation is an irregular supraventricular tachyarrhythmia."),
    ("Rate versus rhythm control",
     "In stable atrial fibrillation, rate control is often an appropriate initial strategy."),
    ("Exceptions", "Rhythm control is preferred in young symptomatic patients."),
    ("CHA2DS2-VASc", "TABLE"),  # 5: table slide
    ("Anticoagulation interruption",
     "Temporary interruption of oral anticoagulant therapy before elective surgery."),
]  # fmt: skip


def make_deck(path, *, variant=0):
    prs = pptx.Presentation()
    for i, (title, body) in enumerate(SLIDES, start=1):
        if body is None:
            s = prs.slides.add_slide(prs.slide_layouts[5])  # title only
            s.shapes.title.text = title
            continue
        s = prs.slides.add_slide(prs.slide_layouts[1])  # title + content
        s.shapes.title.text = title
        if body == "TABLE":
            t = s.shapes.add_table(3, 2, Inches(1), Inches(2), Inches(6), Inches(2)).table
            rows = [("Score", "Annual stroke risk"), ("0", "0.2%"), ("2", "2.2%")]
            for r, (k, v) in enumerate(rows):
                t.cell(r, 0).text, t.cell(r, 1).text = k, v
        else:
            s.placeholders[1].text = body if not (variant and i == 3) else body + " Updated 2027."
        s.notes_slide.notes_text_frame.text = f"Notes for slide {i}."
    prs.save(path)
    return path


def test_extract_diff_and_hashes(tmp_path):
    v1 = pptx_decks.extract_deck(make_deck(tmp_path / "v1.pptx"), course_id="im-2027",
                                 lecture_id="cardio-04", lecture_title="Atrial Fibrillation",
                                 academic_year="2026-2027", version="v1")  # fmt: skip
    assert len(v1) == 6 and all(d.source_type == SourceType.COURSE_SLIDE for d in v1)
    assert v1[0].doc_id == "slide:cardio-04:1" and v1[0].slide.section == ""
    assert v1[1].slide.section == "Atrial Fibrillation"  # inherited from the header slide
    assert "Speaker notes: Notes for slide 2." in v1[1].abstract and v1[1].year == 2026
    assert "Score | Annual stroke risk" in v1[4].abstract and "2 | 2.2%" in v1[4].abstract
    assert not v1[0].is_peer_reviewed and v1[0].course_id == "im-2027"
    v2 = pptx_decks.extract_deck(make_deck(tmp_path / "v2.pptx", variant=1), course_id="im-2027",
                                 lecture_id="cardio-04", lecture_title="Atrial Fibrillation",
                                 academic_year="2027-2028", version="v2")  # fmt: skip
    d = pptx_decks.diff_decks(v1, v2)
    assert d["changed"] == [3] and d["added"] == [] and d["removed"] == [] and d["unchanged"] == 5
    assert d["reindex_doc_ids"] == ["slide:cardio-04:3"]
    out = tmp_path / "v1.jsonl"
    assert pptx_decks.write_jsonl(v1, out) == 6
    assert pptx_decks.read_jsonl(out)[4].slide.slide_number == 5
    rc = pptx_decks.main(["diff", str(out), str(out)])
    assert rc == 0


@pytest.fixture(scope="module")
def mixed_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("mixed")
    deck = make_deck(root / "deck.pptx")
    slides = pptx_decks.extract_deck(deck, course_id="im-2027", lecture_id="cardio-04",
                                     lecture_title="Atrial Fibrillation", version="v1")  # fmt: skip
    other = pptx_decks.extract_deck(deck, course_id="other-course", lecture_id="x-01",
                                    lecture_title="Other", version="v1")  # fmt: skip
    docs = root / "docs.jsonl"
    with open(docs, "w", encoding="utf-8") as fh:
        for d in list(fixture_documents()) + slides + other:
            fh.write(d.model_dump_json() + "\n")
    out = root / "artifact"
    assert build_indexes.main(["--docs", str(docs), "--out", str(out), "--corpus-id", "mixed",
                               "--encoder", "hash:128", "--shard-size", "40"]) == 0  # fmt: skip
    return out


def _retriever(bundle):
    store = MetadataStore(bundle / "metadata.sqlite", read_only=True)
    cfg = RetrievalConfig(k_bm25=50, k_dense=50, rerank_pool=40, top_docs=10)
    return HybridRetriever(store, BM25Index(bundle / "bm25"),
                           DenseIndex(bundle / "dense", HashEncoder(128)), LexicalOverlapReranker(),
                           cfg)  # fmt: skip


def test_course_filter_is_enforced_inside_retrieval(mixed_bundle):
    r = _retriever(mixed_bundle)
    q = "Is rate control an appropriate initial strategy in stable atrial fibrillation?"
    unfiltered = r.retrieve(q)
    assert {d.doc_id.split(":")[1] for d in unfiltered.docs if d.doc_id.startswith("slide:")} >= {
        "cardio-04", "x-01"
    }
    mine = r.retrieve(q, course_ids=["im-2027"])
    slide_docs = [d.doc_id for d in mine.docs if d.doc_id.startswith("slide:")]
    assert slide_docs and all(s.startswith("slide:cardio-04:") for s in slide_docs)
    assert not any(d.doc_id.startswith("pmid:") for d in mine.docs)  # literature is another course
    assert mine.audit.course_filter == ["im-2027"]
    nothing = r.retrieve(q, course_ids=[])
    assert nothing.docs == []  # entitled to nothing


def test_slide_citations_and_neighbour_expansion(mixed_bundle):
    r = _retriever(mixed_bundle)
    res = r.retrieve("rate control initial strategy stable atrial fibrillation",
                     course_ids=["im-2027"])  # fmt: skip
    units = select_evidence(res, r.store, max_units=6, per_doc=1, slide_neighbours=1)
    top = units[0]
    assert top.is_slide and top.citation_id == "SLIDE:cardio-04:3"
    assert top.display_source == "Atrial Fibrillation, Slide 3" and not top.is_peer_reviewed
    neigh = [u for u in units if u.neighbour_of == top.passage_id]
    assert {u.slide_number for u in neigh} == {2, 4}
    rendered = render_evidence(units)
    assert "[SLIDE:cardio-04:3 | 2026 | course_slide | course material" in rendered or \
           "[SLIDE:cardio-04:3 | n.d. | course_slide | course material" in rendered
    assert "Atrial Fibrillation, Slide 3" in rendered
    # the store round-trips slide provenance
    doc = r.store.get_document("slide:cardio-04:5")
    assert doc.slide.slide_title == "CHA2DS2-VASc" and json.loads(doc.slide.model_dump_json())
