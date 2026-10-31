"""Run the API: python -m scripts.serve --bundle data/artifacts/<id> --profile hosted --port 8000"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--host", default=os.environ.get("LITBIORAG_API_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("LITBIORAG_API_PORT", "8471")))
    a = ap.parse_args(argv)
    if a.bundle:
        os.environ["LITBIORAG_BUNDLE"] = str(a.bundle)
    if a.profile:
        os.environ["LITBIORAG_MODEL_PROFILE"] = a.profile
    logging.basicConfig(level=os.environ.get("LITBIORAG_LOG_LEVEL", "INFO"))
    import uvicorn

    from app.api.main import create_app

    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
