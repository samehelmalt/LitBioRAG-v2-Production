Local data lives here and is git-ignored.

```text
data/
├── pubmed/raw/        baseline .xml.gz + .md5 (build host)
├── pubmed/jsonl/      parsed Documents, one .jsonl per baseline file (build host)
├── preprints/         <server>.raw.jsonl, <server>.cursor, <server>.jsonl (build host)
└── artifacts/<id>/    bundles: manifest.json, metadata.sqlite, bm25/, dense/ (both hosts)
```

See `docs/CORPUS.md` for ingestion and `docs/DEPLOYMENT.md` for moving bundles between hosts.
