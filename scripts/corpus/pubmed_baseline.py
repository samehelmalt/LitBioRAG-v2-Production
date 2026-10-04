"""PubMed annual baseline -> normalized JSONL documents.

Runs on the BUILD host. Three resumable sub-commands:

    python -m scripts.corpus.pubmed_baseline list --out data/pubmed/files.txt
    python -m scripts.corpus.pubmed_baseline download --files data/pubmed/files.txt \
        --dest data/pubmed/raw
    python -m scripts.corpus.pubmed_baseline parse --src data/pubmed/raw \
        --dest data/pubmed/jsonl [--min-year 2015] [--require-abstract]

* ``list`` reads the NCBI baseline directory and writes one ``pubmedNNnNNNN.xml.gz`` name per line.
* ``download`` fetches each file plus its ``.md5``; a file whose md5 already matches is skipped.
* ``parse`` streams each ``.xml.gz`` with ``iterparse`` (constant memory) and writes
  ``<name>.jsonl`` with one ``Document`` per line; an existing output file means the input is done.

Only abstracts and metadata are kept; full text is out of scope for v2 Step 3.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.core.documents import Document, source_type_from_pubtypes  # noqa: E402

BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
_NAME_RE = re.compile(r"pubmed\d+n\d+\.xml\.gz")


# ---------------------------------------------------------------- listing / download ----------
def list_baseline_files(url: str = BASELINE_URL) -> list[str]:
    with urllib.request.urlopen(url, timeout=60) as r:
        html = r.read().decode("utf-8", errors="ignore")
    return sorted(set(_NAME_RE.findall(html)))


def md5_of(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - NCBI publishes md5 checksums
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(name: str, dest: Path, url: str = BASELINE_URL) -> bool:
    """Download ``name`` and its .md5 into ``dest``; return True if a download happened."""
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / name
    with urllib.request.urlopen(f"{url}{name}.md5", timeout=60) as r:
        expected = r.read().decode().strip().split()[-1]
    if target.exists() and md5_of(target) == expected:
        return False
    tmp = target.with_suffix(".part")
    with urllib.request.urlopen(f"{url}{name}", timeout=600) as r, open(tmp, "wb") as fh:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            fh.write(chunk)
    if md5_of(tmp) != expected:
        tmp.unlink(missing_ok=True)
        raise OSError(f"md5 mismatch for {name}")
    tmp.replace(target)
    return True


# ---------------------------------------------------------------- parsing ---------------------
def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _abstract(article: ET.Element) -> str:
    parts = []
    for at in article.findall("./Abstract/AbstractText"):
        label = at.get("Label")
        body = _text(at)
        if not body:
            continue
        labelled = label and label.upper() != "UNLABELLED"
        parts.append(f"{label.title()}: {body}" if labelled else body)
    return " ".join(parts).strip()


def _year(article: ET.Element) -> int | None:
    pd = article.find("./Journal/JournalIssue/PubDate")
    if pd is None:
        return None
    y = pd.findtext("Year")
    if y and y.isdigit():
        return int(y)
    m = re.search(r"\b(19|20)\d{2}\b", pd.findtext("MedlineDate") or "")
    return int(m.group(0)) if m else None


def parse_article(el: ET.Element) -> Document | None:
    """Convert one <PubmedArticle> element to a Document; None if it has no title."""
    cit = el.find("MedlineCitation")
    if cit is None:
        return None
    pmid = cit.findtext("PMID")
    article = cit.find("Article")
    if article is None or not pmid:
        return None
    title = _text(article.find("ArticleTitle")).rstrip(".")
    if not title:
        return None
    pub_types = [_text(p) for p in article.findall("./PublicationTypeList/PublicationType")]
    mesh = [_text(m) for m in cit.findall("./MeshHeadingList/MeshHeading/DescriptorName")]
    authors = []
    for a in article.findall("./AuthorList/Author"):
        last, fore = a.findtext("LastName"), a.findtext("ForeName")
        if last:
            authors.append(f"{last} {fore}".strip())
        elif a.findtext("CollectiveName"):
            authors.append(a.findtext("CollectiveName"))
    doi = None
    for aid in el.findall("./PubmedData/ArticleIdList/ArticleId"):
        if aid.get("IdType") == "doi" and aid.text:
            doi = aid.text.strip().lower()
    return Document(
        doc_id=Document.make_id(pmid, None),
        pmid=pmid,
        doi=doi,
        title=title,
        abstract=_abstract(article),
        source_type=source_type_from_pubtypes(pub_types),
        year=_year(article),
        journal=article.findtext("./Journal/Title"),
        pub_types=pub_types,
        mesh_terms=mesh,
        authors=authors,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        source="pubmed",
    )


def iter_documents(xml_gz: Path) -> Iterator[Document]:
    """Stream <PubmedArticle> elements from a baseline file with constant memory."""
    with gzip.open(xml_gz, "rb") as fh:
        for _, el in ET.iterparse(fh, events=("end",)):
            if el.tag == "PubmedArticle":
                doc = parse_article(el)
                if doc is not None:
                    yield doc
                el.clear()


def parse_file(
    xml_gz: Path, dest: Path, *, min_year: int | None = None, require_abstract: bool = False
) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / (xml_gz.name.replace(".xml.gz", "") + ".jsonl")
    if out.exists():
        return {"file": xml_gz.name, "skipped": True}
    tmp = out.with_suffix(".part")
    kept = seen = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        for doc in iter_documents(xml_gz):
            seen += 1
            if min_year and (doc.year is None or doc.year < min_year):
                continue
            if require_abstract and not doc.abstract:
                continue
            fh.write(doc.model_dump_json() + "\n")
            kept += 1
    tmp.replace(out)
    return {"file": xml_gz.name, "seen": seen, "kept": kept, "skipped": False}


# ---------------------------------------------------------------- CLI -------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--url", default=BASELINE_URL)
    p = sub.add_parser("download")
    p.add_argument("--files", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--url", default=BASELINE_URL)
    p = sub.add_parser("parse")
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--min-year", type=int, default=None)
    p.add_argument("--require-abstract", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        names = list_baseline_files(a.url)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("\n".join(names) + "\n")
        print(f"{len(names)} files -> {a.out}")
    elif a.cmd == "download":
        names = [n.strip() for n in a.files.read_text().splitlines() if n.strip()]
        got = sum(download_file(n, a.dest, a.url) for n in names)
        print(f"downloaded {got}, already present {len(names) - got}")
    elif a.cmd == "parse":
        for f in sorted(a.src.glob("*.xml.gz")):
            print(json.dumps(parse_file(f, a.dest, min_year=a.min_year,
                                        require_abstract=a.require_abstract)))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
