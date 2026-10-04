import gzip
import json

from app.core.documents import SourceType
from scripts.corpus import preprints, pubmed_baseline

PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
<PubmedArticle>
 <MedlineCitation Status="MEDLINE" Owner="NLM">
  <PMID Version="1">26561343</PMID>
  <Article PubModel="Print">
   <Journal><Title>Haemophilia</Title>
    <JournalIssue><PubDate><Year>2016</Year><Month>Jan</Month></PubDate></JournalIssue></Journal>
   <ArticleTitle>Coated platelets and severe haemophilia A bleeding phenotype.</ArticleTitle>
   <Abstract>
    <AbstractText Label="BACKGROUND">Coated platelets are a subset of platelets.</AbstractText>
    <AbstractText Label="RESULTS">Levels were 23.5% vs. 41.2% (p &lt; 0.01).</AbstractText>
   </Abstract>
   <AuthorList><Author><LastName>Smith</LastName><ForeName>A B</ForeName></Author>
    <Author><CollectiveName>The Study Group</CollectiveName></Author></AuthorList>
   <PublicationTypeList><PublicationType>Journal Article</PublicationType>
    <PublicationType>Observational Study</PublicationType></PublicationTypeList>
  </Article>
  <MeshHeadingList><MeshHeading><DescriptorName>Hemophilia A</DescriptorName></MeshHeading>
  </MeshHeadingList>
 </MedlineCitation>
 <PubmedData><ArticleIdList><ArticleId IdType="doi">10.1111/HAE.12823</ArticleId>
  <ArticleId IdType="pubmed">26561343</ArticleId></ArticleIdList></PubmedData>
</PubmedArticle>
<PubmedArticle>
 <MedlineCitation Status="MEDLINE" Owner="NLM">
  <PMID Version="1">1</PMID>
  <Article><Journal><JournalIssue><PubDate><MedlineDate>1998 Dec-1999 Jan</MedlineDate>
   </PubDate></JournalIssue></Journal>
   <ArticleTitle>Old paper without abstract</ArticleTitle>
   <PublicationTypeList><PublicationType>Letter</PublicationType></PublicationTypeList>
  </Article>
 </MedlineCitation>
</PubmedArticle>
</PubmedArticleSet>
"""


def test_pubmed_parse_and_filters(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    with gzip.open(src / "pubmed25n0001.xml.gz", "wt", encoding="utf-8") as fh:
        fh.write(PUBMED_XML)
    docs = list(pubmed_baseline.iter_documents(src / "pubmed25n0001.xml.gz"))
    assert len(docs) == 2
    d = docs[0]
    assert d.doc_id == "pmid:26561343" and d.doi == "10.1111/hae.12823"
    assert d.title == "Coated platelets and severe haemophilia A bleeding phenotype"
    assert d.abstract.startswith("Background: Coated platelets") and "23.5%" in d.abstract
    assert d.source_type == SourceType.OBSERVATIONAL and d.year == 2016
    assert d.mesh_terms == ["Hemophilia A"] and d.authors == ["Smith A B", "The Study Group"]
    assert docs[1].year == 1998 and docs[1].source_type == SourceType.OTHER

    out = tmp_path / "jsonl"
    r = pubmed_baseline.parse_file(
        src / "pubmed25n0001.xml.gz", out, min_year=2015, require_abstract=True
    )
    assert r["seen"] == 2 and r["kept"] == 1
    lines = (out / "pubmed25n0001.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["pmid"] == "26561343"
    # resumable: second call skips
    assert pubmed_baseline.parse_file(src / "pubmed25n0001.xml.gz", out)["skipped"] is True


def test_baseline_listing_regex():
    html = '<a href="pubmed25n0002.xml.gz">x</a> <a href="pubmed25n0002.xml.gz.md5">y</a> junk'
    assert pubmed_baseline._NAME_RE.findall(html) == ["pubmed25n0002.xml.gz"] * 2


def _fake_api(total: int):
    recs = [
        {
            "doi": f"10.1101/2024.01.{i:02d}",
            "title": f"Preprint {i}.",
            "authors": "Doe, J.; Roe, R.",
            "date": "2024-01-15",
            "version": "1" if i % 2 else "2",
            "category": "neuroscience",
            "abstract": f"Abstract {i}.",
            "server": "medrxiv",
        }
        for i in range(total)
    ]
    # a later version of record 0 appears again
    recs.append(dict(recs[0], version="3", abstract="Updated abstract 0."))

    def fetch(server, start, end, cursor):
        page = recs[cursor : cursor + preprints.PAGE]
        return {
            "messages": [{"status": "ok", "total": len(recs), "cursor": cursor}],
            "collection": page,
        }

    return fetch, recs


def test_preprints_fetch_resume_and_dedupe(tmp_path, monkeypatch):
    fetch, recs = _fake_api(total=150)
    r1 = preprints.fetch_all(
        "medrxiv", "2024-01-01", "2024-02-01", tmp_path, fetch=fetch, sleep_s=0, max_pages=1
    )
    assert r1["cursor"] == 100 and (tmp_path / "medrxiv.cursor").read_text() == "100"
    r2 = preprints.fetch_all(
        "medrxiv", "2024-01-01", "2024-02-01", tmp_path, fetch=fetch, sleep_s=0
    )
    assert r2["cursor"] == len(recs)
    raw_lines = (tmp_path / "medrxiv.raw.jsonl").read_text().splitlines()
    assert len(raw_lines) == len(recs)

    res = preprints.write_documents("medrxiv", tmp_path)
    assert res["dois"] == 150 and res["documents"] == 150
    docs = [json.loads(line) for line in (tmp_path / "medrxiv.jsonl").read_text().splitlines()]
    d0 = next(d for d in docs if d["doi"] == "10.1101/2024.01.00")
    assert d0["version"] == "3" and d0["abstract"] == "Updated abstract 0."
    assert d0["source_type"] == "preprint" and d0["doc_id"] == "doi:10.1101/2024.01.00"
    assert d0["authors"] == ["Doe, J.", "Roe, R."] and d0["year"] == 2024
