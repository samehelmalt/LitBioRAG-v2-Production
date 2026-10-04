CI lives in `.github/workflows/ci.yml` (ruff + pytest on every push and pull request). Regression gates (false-PASS, retrieval recall, latency) are added here once `evaluation/` produces a baseline.
