# Thesis baseline results (read-only copy)

Copied verbatim from `samehelmalt/LitBioRAG` at commit `ad0bacd0f3b878cd90f9a473bc79018223b27f1b` on 2026-10-04.

| Here | Origin |
|---|---|
| `V5521_Run/thesis_ladder_*.csv` (18 files) | `Results/V5521_Run/` — master run, 12–13 June 2026, engine v5.5.21, two-stage cascade path |
| `V5521_Run/gate_sensitivity_report.txt` | `Results/V5521_Run/` |
| `calib/calib_1st_run/` | `Results/calib/calib_1st_run/` — one-stage CE, development thresholds 0.28/0.18, same 24 queries |
| `calib/calib_2nd_run/` | `Results/calib/calib_2nd_run/` — one-stage CE, 30 held-out calibration queries (`benchmark_heldout_S7.csv`) and re-test with thresholds 0.42/0.30 |

These files are the regression reference for every production run. They are never edited here; corrections or re-analyses are written as new files elsewhere and cite these.
