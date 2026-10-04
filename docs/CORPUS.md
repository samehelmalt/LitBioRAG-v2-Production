# Corpus

## Scope (owner decision 2026-10-04)

- **PubMed**: the full annual baseline, abstracts and metadata only. Step 3 starts with the window **2015–present** so a usable bundle exists within hours on the build host; the pre-2015 backfill is the same scripts with `--min-year` removed and runs as a resumable job.
- **Preprints**: bioRxiv and medRxiv via `api.biorxiv.org`, latest version per DOI, `source_type = preprint`. Preprints rank below every peer-reviewed source class in the evidence hierarchy (`app/core/documents.py: EVIDENCE_RANK`) and are labelled as non-peer-reviewed wherever evidence is shown.
- Not in scope for Step 3: PMC full text, other databases, non-English records.

## Record model

`Document` (`app/core/documents.py`): `doc_id` (`pmid:<n>` or `doi:<doi>`), title, abstract, `source_type`, pmid, doi, year, journal, pub_types, mesh_terms, authors, url, source (`pubmed|biorxiv|medrxiv`), version.
`source_type` comes from PubMed publication types (systematic review / meta-analysis → `systematic_review`; RCT; clinical trial; observational; case report; review; journal article; otherwise `other`); preprints are always `preprint`.

`Passage`: sentence-aligned chunks of `title + "\n" + abstract`, target 200 words, max 300, one-sentence overlap, with `char_start/char_end` offsets into the document text (`app/retrieval/passages.py`). Most abstracts yield 1–3 passages. Every evidence span shown to a user or a verifier is traceable to a passage id and an offset.

## Ingestion (build host)

```bash
# PubMed baseline
python -m scripts.corpus.pubmed_baseline list     --out data/pubmed/files.txt
python -m scripts.corpus.pubmed_baseline download --files data/pubmed/files.txt --dest data/pubmed/raw
python -m scripts.corpus.pubmed_baseline parse    --src data/pubmed/raw --dest data/pubmed/jsonl \
    --min-year 2015 --require-abstract

# Preprints
python -m scripts.corpus.preprints fetch  --server biorxiv --from 2013-11-01 --to 2026-10-01 --dest data/preprints
python -m scripts.corpus.preprints fetch  --server medrxiv --from 2019-06-01 --to 2026-10-01 --dest data/preprints
python -m scripts.corpus.preprints dedupe --server biorxiv --dest data/preprints
python -m scripts.corpus.preprints dedupe --server medrxiv --dest data/preprints
```

All three stages are resumable: downloaded files are md5-verified and skipped when present, parsed files are skipped when their `.jsonl` exists, preprint paging continues from `<server>.cursor`.

Expected sizes (order of magnitude, from NCBI and bioRxiv public counts): PubMed 2015–present ≈ 16–18 M records with abstracts (≈ 1,300 baseline files, ≈ 40 GB compressed XML); full baseline ≈ 37 M records; bioRxiv + medRxiv ≈ 300 k preprints. These are planning figures, to be replaced by the counts in the bundle manifest after the first build.

## Index build (build host)

```bash
python -m scripts.build_indexes \
    --docs "data/pubmed/jsonl/*.jsonl" "data/preprints/*.jsonl" \
    --out artifacts/pubmed2015plus_v1 --corpus-id pubmed2015plus_v1 \
    --encoder hf:<model id chosen in docs/MODEL_SELECTION.md> --batch-size 128 --shard-size 100000
```

Stages (`store → bm25 → embed → dense → pack`) leave `.<stage>.done` markers and are skipped on re-run; embedding is sharded so a crash resumes at the first missing shard. The encoder id, revision and dimension are written to the manifest; the serving host refuses an encoder that does not match.

Until a dense encoder has been selected (Step 5), bundles can be built with `--encoder hash:256` to exercise BM25, the store and the API end to end. Such a bundle is for plumbing only and must never be used for a reported number.

## Bundle layout

```text
artifacts/<corpus_id>/
├── manifest.json        corpus_id, created_at, counts, encoder, dense params, sha256 per file
├── metadata.sqlite      documents + passages (read-only on the serving host)
├── bm25/                Tantivy index (on disk)
└── dense/
    ├── index.faiss      FlatIP (< 50k passages) or IVF-PQ, memory-mappable
    ├── ids.npy          passage id per vector row
    ├── meta.json        encoder, dim, n, index params
    └── encoder.json
```

## Known limitations recorded for the failure analysis

- Abstract-only: claims that live in full-text results sections cannot be verified from this corpus.
- PubMed publication types are incomplete for recent records (MEDLINE indexing lag); `source_type` for those falls to `journal_article`.
- Preprint abstracts from the API occasionally contain JATS markup fragments; the parser keeps text as delivered. Cleaning is a Step 4 item if it shows up in retrieval failures.
