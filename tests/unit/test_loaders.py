from evaluation.datasets.loaders import (
    load_all,
    load_bioasq,
    load_pubmedqa,
    load_thesis_queries,
    normalize_pmid,
    normalize_question,
    query_id,
)


def test_normalize_pmid_variants():
    assert normalize_pmid("http://www.ncbi.nlm.nih.gov/pubmed/25731175") == "25731175"
    assert normalize_pmid("PMID: 123") == "123"
    assert normalize_pmid(26561343) == "26561343"
    assert normalize_pmid("") == ""


def test_normalize_question_is_stable():
    a = normalize_question("Is the Wnt protein modified by Notum?")
    b = normalize_question("is the wnt  protein modified by notum")
    assert a == b
    assert query_id("Is the Wnt protein modified by Notum?", "BioASQ") == query_id(
        "is the wnt protein modified by notum", "BioASQ"
    )


def test_thesis_24():
    recs = load_thesis_queries()
    assert len(recs) == 24
    assert all(r.gold_pmids for r in recs)


def test_extracted_sets_have_gold_pmids_and_types():
    b, p = load_bioasq(), load_pubmedqa()
    assert len(b) == 1562 and len(p) == 2000
    assert all(r.gold_pmids for r in b + p)
    assert {r.question_type for r in b} <= {"yesno", "factoid", "list", "summary", "unknown"}
    assert any(r.yn_label == "yes" for r in b)


def test_load_all_dedupes_and_contains_thesis_queries():
    pool = load_all()
    norms = {normalize_question(r.question) for r in pool}
    assert len(norms) == len(pool)
    hits = sum(normalize_question(t.question) in norms for t in load_thesis_queries())
    assert hits == 24
