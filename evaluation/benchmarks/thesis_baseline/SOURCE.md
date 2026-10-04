# Thesis baseline results (read-only copy)

Copied verbatim from `samehelmalt/LitBioRAG` at commit `ad0bacd0f3b878cd90f9a473bc79018223b27f1b` on 2026-10-04.

| Here | Origin |
|---|---|
| `V5521_Run/thesis_ladder_*.csv` (18 files) | `Results/V5521_Run/` — master run, 12–13 June 2026, engine v5.5.21, two-stage cascade path |
| `V5521_Run/gate_sensitivity_report.txt` | `Results/V5521_Run/` |
| `calib/calib_1st_run/` | `Results/calib/calib_1st_run/` — one-stage CE, S7 only, **30 held-out calibration queries** (540 rows; disjoint from the 24 by text and gold PMID); `benchmark_24_queries.json` lists the 24 evaluation queries |
| `calib/calib_2nd_run/` | `Results/calib/calib_2nd_run/` — one-stage CE, S7 only, re-test of the **24 evaluation queries** with the thresholds derived on the 30 (`benchmark_heldout_S7.csv` is the 432-row concatenation) |

These files are the regression reference for every production run. They are never edited here; corrections or re-analyses are written as new files elsewhere and cite these.
