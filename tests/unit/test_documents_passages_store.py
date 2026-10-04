import sqlite3

import pytest

from app.core.documents import Document, SourceType, source_type_from_pubtypes
from app.retrieval.passages import split_document, split_sentences
from app.retrieval.store import MetadataStore


def make_doc(abstract: str, **kw) -> Document:
    base = dict(
        doc_id="pmid:1", pmid="1", title="A title.", abstract=abstract,
        source_type=SourceType.JOURNAL_ARTICLE, year=2020,
    )  # fmt: skip
    base.update(kw)
    return Document(**base)


def test_source_type_mapping_and_rank():
    assert source_type_from_pubtypes(["Journal Article", "Meta-Analysis"]) == (
        SourceType.SYSTEMATIC_REVIEW
    )
    assert source_type_from_pubtypes(["Journal Article", "Review"]) == SourceType.REVIEW
    assert source_type_from_pubtypes(["Journal Article"]) == SourceType.JOURNAL_ARTICLE
    assert source_type_from_pubtypes(["Letter"]) == SourceType.OTHER
    pre = Document(doc_id="doi:10.1/x", doi="10.1/X", title="t", abstract="a",
                   source_type=SourceType.PREPRINT, source="biorxiv")  # fmt: skip
    assert not pre.is_peer_reviewed and pre.doi == "10.1/x"
    assert pre.evidence_rank > make_doc("a").evidence_rank
    assert Document.make_id("PMID 123", None) == "pmid:123"
    with pytest.raises(ValueError):
        Document.make_id(None, None)


def test_sentence_split_protects_abbreviations_and_decimals():
    text = "Mortality fell vs. placebo (p = 0.05). Fig. 2 shows this. Dose was 2.5 mg. Done."
    spans = split_sentences(text)
    sents = [text[s:e] for s, e in spans]
    assert sents == [
        "Mortality fell vs. placebo (p = 0.05).",
        "Fig. 2 shows this.",
        "Dose was 2.5 mg.",
        "Done.",
    ]


def test_passages_cover_text_with_offsets_and_overlap():
    sentences = [f"Sentence number {i} has exactly seven words here." for i in range(30)]
    doc = make_doc(" ".join(sentences))
    ps = list(split_document(doc, target_words=50, max_words=60, overlap_sentences=1))
    assert len(ps) > 1
    for p in ps:
        assert doc.text[p.char_start : p.char_end] == p.text
        assert p.n_words <= 60
        assert p.passage_id == f"{doc.doc_id}#{p.ord}"
    # overlap: the last sentence of passage k is the first of passage k+1
    for a, b in zip(ps, ps[1:], strict=False):
        assert b.text.startswith(a.text.split(". ")[-1].rstrip(".")[:20])
    # passage 0 carries the title
    assert ps[0].text.startswith("A title.")
    assert ps[-1].char_end == len(doc.text)


def test_single_short_document_is_one_passage():
    doc = make_doc("Short abstract.")
    ps = list(split_document(doc))
    assert len(ps) == 1 and ps[0].text == doc.text


def test_store_roundtrip(tmp_path):
    store = MetadataStore(tmp_path / "m.sqlite")
    d1 = make_doc("Alpha beta. Gamma delta.", pub_types=["Journal Article"], mesh_terms=["X"])
    d2 = Document(doc_id="doi:10.1101/2024.01.01", doi="10.1101/2024.01.01", title="Pre",
                  abstract="Preprint text.", source_type=SourceType.PREPRINT, source="medrxiv",
                  version="2")  # fmt: skip
    store.add_documents([d1, d2])
    ps = list(split_document(d1)) + list(split_document(d2))
    store.add_passages(ps)
    store.set_meta("corpus_id", "test")
    assert store.counts() == {"documents": 2, "passages": len(ps)}
    assert store.get_by_pmid("PMID: 1") == d1
    assert store.get_by_doi("10.1101/2024.01.01") == d2
    assert store.get_passage(ps[0].passage_id) == ps[0]
    assert set(store.get_passages([p.passage_id for p in ps])) == {p.passage_id for p in ps}
    assert store.passages_of(d1.doc_id)[0].ord == 0
    assert store.pmids() == {"1"}
    assert store.get_meta("corpus_id") == "test"
    assert len(list(store.iter_passages(batch=1))) == len(ps)
    store.close()
    ro = MetadataStore(tmp_path / "m.sqlite", read_only=True)
    assert ro.get_document("pmid:1") == d1
    with pytest.raises(sqlite3.OperationalError):
        ro.set_meta("x", "y")
