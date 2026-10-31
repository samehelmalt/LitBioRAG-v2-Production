# Corpus

## Scope (owner decision 2026-10-04)

- **PubMed**: the full annual baseline, abstracts and metadata only. Step 3 starts with the window **2015–present** so a usable bundle exists within hours on the build host; the pre-2015 backfill is the same scripts with `--min-year` removed and runs as a resumable job.
- **Preprints**: bioRxiv and medRxiv via `api.biorxiv.org`, latest version per DOI, `source_type = preprint`. Preprints rank below every peer-reviewed source class in the evidence hierarchy (`app/core/documents.py: EVIDENCE_RANK`) and are labelled as non-peer-reviewed wherever evidence is shown.
- **Course material (education product)**: PowerPoint decks via `scripts/corpus/pptx_decks.py`, one `Document` per slide with `source_type = course_slide` and slide provenance (`SlideMeta`); ranked as course material, never as peer-reviewed literature; filtered by `course_id` inside retrieval for entitlements. Annual updates: `pptx_decks diff` lists changed slides for incremental re-index.
- Not in scope for Step 3: PMC full text, other databases, non-English records, OCR of slide images (hook planned).

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

## Uploaded knowledge bases (`app/kb/`)
Besides the served corpus bundle (PubMed, bioRxiv/medRxiv, course decks), users can upload their
own material: PDF, DOCX, XLSX, PPTX, TXT and Markdown. Each file becomes citable units — a page, a
heading section, a block of spreadsheet rows (with its cell range), a slide — stored with full
provenance (`FileMeta`: knowledge base, file id = content hash, locator, label, version). Units
are indexed incrementally into a per-server *uploads bundle* (SQLite + Tantivy + FAISS Flat) that
is merged with the main corpus at query time by RRF and reranked once. Entitlement: the owner,
users the owner shared with, and admins; the filter is applied inside retrieval, before fusion.

**Retrieval scope** on every query: `indexed` (served corpus only), `uploaded` (the caller's
knowledge bases only), or `both` (default from `configs/api.yaml`). Uploaded units are tier
"other" (provenance unknown to the system) and are never shown as peer-reviewed. Scanned PDFs
yield no text until an OCR hook is configured (`PdfAdapter(ocr=...)`).

## Managing the served indexes (`app/corpus/manager.py`, admin UI tab "Corpus Indexes")
The serving host never rebuilds the full PubMed index; it **fetches** a bundle built on the build
host (Drive/URL/path), **verifies** checksums, **activates** it (retrieval swaps in place, models
stay loaded), and **updates** it incrementally: PubMed daily update files or a bioRxiv/medRxiv
window are parsed to JSONL and appended with `scripts/update_bundle.py` (store + BM25 + dense
`add`; replaced documents are counted as `stale_vectors` in the manifest until the next full
rebuild). A **rebuild** request on the serving host is refused and the exact build-host command is
shown instead (`allow_build_on_serve: true` on a build host). The **indexed corpus can be disabled**
so that every query is answered from uploaded knowledge bases only; the default scope is `both`.
