"""bioRxiv / medRxiv -> normalized JSONL documents (source_type = preprint).

Runs on the BUILD host, resumable by cursor:

    python -m scripts.corpus.preprints fetch --server medrxiv --from 2019-06-01 --to 2026-10-01 \
        --dest data/preprints

The public API (https://api.biorxiv.org/details/{server}/{from}/{to}/{cursor}/json) returns 100
records per page. Every page is appended to ``<server>.raw.jsonl`` and the next cursor is written
to ``<server>.cursor``; re-running continues from there. ``dedupe`` then keeps the latest version
of each DOI and writes ``<server>.jsonl`` of Documents. No API key is needed; be polite (one
request at a time).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.core.documents import Document, SourceType  # noqa: E402

API = "https://api.biorxiv.org/details/{server}/{start}/{end}/{cursor}/json"
SERVERS = ("biorxiv", "medrxiv")
PAGE = 100


def fetch_page(server: str, start: str, end: str, cursor: int, timeout: int = 60) -> dict:
    url = API.format(server=server, start=start, end=end, cursor=cursor)
    req = urllib.request.Request(url, headers={"User-Agent": "LitBioRAG-v2 corpus builder"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_all(
    server: str,
    start: str,
    end: str,
    dest: Path,
    *,
    fetch: Callable[..., dict] = fetch_page,
    sleep_s: float = 0.5,
    max_pages: int | None = None,
) -> dict:
    """Append raw records to <server>.raw.jsonl, resuming from <server>.cursor."""
    if server not in SERVERS:
        raise ValueError(f"server must be one of {SERVERS}")
    dest.mkdir(parents=True, exist_ok=True)
    raw = dest / f"{server}.raw.jsonl"
    cursor_file = dest / f"{server}.cursor"
    cursor = int(cursor_file.read_text()) if cursor_file.exists() else 0
    pages = 0
    total = None
    with open(raw, "a", encoding="utf-8") as fh:
        while True:
            data = fetch(server, start, end, cursor)
            msg = (data.get("messages") or [{}])[0]
            coll = data.get("collection") or []
            total = int(msg.get("total", total or 0) or 0)
            for rec in coll:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cursor += len(coll)
            cursor_file.write_text(str(cursor))
            pages += 1
            done = not coll or len(coll) < PAGE or cursor >= total
            if done or (max_pages and pages >= max_pages):
                break
            time.sleep(sleep_s)
    return {"server": server, "cursor": cursor, "total": total, "pages": pages}


def to_document(rec: dict, server: str) -> Document | None:
    doi = (rec.get("doi") or "").strip().lower()
    title = (rec.get("title") or "").strip().rstrip(".")
    if not doi or not title:
        return None
    date = rec.get("date") or ""
    year = int(date[:4]) if date[:4].isdigit() else None
    authors = [a.strip() for a in (rec.get("authors") or "").split(";") if a.strip()]
    return Document(
        doc_id=Document.make_id(None, doi),
        doi=doi,
        title=title,
        abstract=(rec.get("abstract") or "").strip(),
        source_type=SourceType.PREPRINT,
        year=year,
        journal=server,
        pub_types=["Preprint", str(rec.get("category") or "").strip()],
        authors=authors,
        url=f"https://doi.org/{doi}",
        source=server,
        version=str(rec.get("version") or "") or None,
    )


def dedupe_latest(records: Iterable[dict]) -> dict[str, dict]:
    """Keep the highest version per DOI."""
    best: dict[str, dict] = {}
    for rec in records:
        doi = (rec.get("doi") or "").strip().lower()
        if not doi:
            continue
        try:
            v = int(rec.get("version") or 0)
        except ValueError:
            v = 0
        if doi not in best or v > int(best[doi].get("version") or 0):
            best[doi] = rec
    return best


def write_documents(server: str, dest: Path) -> dict:
    raw = dest / f"{server}.raw.jsonl"
    out = dest / f"{server}.jsonl"
    with open(raw, encoding="utf-8") as fh:
        recs = (json.loads(line) for line in fh if line.strip())
        latest = dedupe_latest(recs)
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for rec in latest.values():
            doc = to_document(rec, server)
            if doc is not None:
                fh.write(doc.model_dump_json() + "\n")
                n += 1
    return {"server": server, "dois": len(latest), "documents": n, "out": str(out)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--server", choices=SERVERS, required=True)
    p.add_argument("--from", dest="start", required=True, help="YYYY-MM-DD")
    p.add_argument("--to", dest="end", required=True, help="YYYY-MM-DD")
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--max-pages", type=int, default=None)
    p = sub.add_parser("dedupe")
    p.add_argument("--server", choices=SERVERS, required=True)
    p.add_argument("--dest", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        print(json.dumps(fetch_all(a.server, a.start, a.end, a.dest, max_pages=a.max_pages)))
    else:
        print(json.dumps(write_documents(a.server, a.dest)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
