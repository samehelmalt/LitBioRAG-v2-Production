"""Run the adversarial suites against a bundle with the models of a registry profile.

    python -m scripts.run_adversarial --bundle data/artifacts/<id> --profile default \
        --out evaluation/benchmarks/runs/<name>/adversarial

Currently: the near-neighbour substitution suite. Exit code 1 if the catch rate is below 1.0 or
the false-alarm rate above the allowed level, so CI and release checks can gate on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval.store import MetadataStore  # noqa: E402
from app.verification.pipeline import VerificationConfig  # noqa: E402
from evaluation.adversarial.substitution import build_cases, run_cases, write_run  # noqa: E402
from models.registry import ModelRegistry  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=200)
    ap.add_argument("--max-false-alarm", type=float, default=0.0)
    a = ap.parse_args(argv)

    reg = ModelRegistry.load(profile=a.profile)
    store = MetadataStore(a.bundle / "metadata.sqlite", read_only=True)
    docs = [store.get_document(r[0]) for r in store.conn.execute("SELECT doc_id FROM documents")]
    cases = build_cases(docs, max_cases=a.max_cases)
    by_id = {d.doc_id: d for d in docs}
    results = run_cases(
        cases, by_id, reg.build("verifier"), reg.build("entity_extractor"),
        ver_cfg=VerificationConfig.from_yaml(ROOT / "configs" / "verification.yaml"),
    )  # fmt: skip
    s = write_run(cases, results, a.out)
    s["models"] = reg.describe()
    print(json.dumps({k: v for k, v in s.items() if k != "models"}, indent=1))
    ok = (s["catch_rate"] in (None, 1.0)) and (s["false_alarm_rate"] or 0.0) <= a.max_false_alarm
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
