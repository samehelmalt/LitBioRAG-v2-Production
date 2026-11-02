"""Concurrency load test against a running AEOS Medical API.

    python -m scripts.load_test --base-url http://127.0.0.1:8471 --token "$TOKEN" \
        --users 10 50 100 --requests-per-user 5 --questions data/prototype/questions.jsonl \
        --repeat-ratio 0.5 --out evaluation/benchmarks/runs/<name>/load

For each user count it opens that many concurrent clients, each asking ``requests-per-user``
questions back to back, and reports per level: throughput (answers/min), latency p50/p95/p99,
status counts (200 / 429 / 503 / other), the Retry-After seen on 503s, generator tokens per
second, cache-hit share (``usage.cached``) and the delivery-action mix. Questions are drawn from
the file; ``repeat-ratio`` is the share of asks that repeat an earlier question (students ask the
same things, and the generation cache serves repeats without the generator).

Numbers from a run on test doubles measure the service overhead only; capacity claims in
docs/DEPLOYMENT.md are filled from a run against a real profile on the serving host. The script
needs ``httpx`` (``api`` extra).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_questions(path: Path | None) -> list[str]:
    if path is None:
        return ["What is the first-line treatment of severe hyperkalaemia?",
                "Which ECG changes are seen with hyperkalaemia?",
                "How does insulin lower serum potassium?",
                "What causes hypokalaemia in diuretic therapy?",
                "What does MAP3K3 do in NF-kappaB signalling?"]
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            q = json.loads(line).get("question")
            if q:
                out.append(q)
        else:
            out.append(line)
    if not out:
        raise SystemExit(f"no questions in {path}")
    return out


def pick_questions(pool: list[str], n: int, repeat_ratio: float, rng: random.Random) -> list[str]:
    asked: list[str] = []
    for _ in range(n):
        if asked and rng.random() < repeat_ratio:
            asked.append(rng.choice(asked))
        else:
            asked.append(rng.choice(pool))
    return asked


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return round(s[k], 1)


async def _one_user(client, uid: int, questions: list[str], scope: str | None,
                    results: list[dict[str, Any]]) -> None:  # fmt: skip
    for q in questions:
        body: dict[str, Any] = {"question": q}
        if scope:
            body["scope"] = scope
        t0 = time.perf_counter()
        rec: dict[str, Any] = {"user": uid, "question": q}
        try:
            r = await client.post("/api/v1/query", json=body)
            rec["status"] = r.status_code
            rec["ms"] = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                js = r.json()
                u = js.get("usage") or {}
                rec["action"] = js.get("action")
                rec["cached"] = bool(u.get("cached"))
                rec["completion_tokens"] = int(u.get("completion_tokens") or 0)
                rec["prompt_tokens"] = int(u.get("prompt_tokens") or 0)
                rec["timings_ms"] = js.get("timings_ms") or {}
            elif r.status_code in (429, 503):
                rec["retry_after"] = r.headers.get("Retry-After")
        except Exception as e:  # noqa: BLE001  (a failed call is a data point, not a crash)
            rec["status"] = 0
            rec["ms"] = (time.perf_counter() - t0) * 1000
            rec["error"] = type(e).__name__
        results.append(rec)


async def run_level(make_client, users: int, per_user: int, pool: list[str], repeat_ratio: float,
                    scope: str | None, seed: int) -> dict[str, Any]:  # fmt: skip
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    async with make_client() as client:
        await asyncio.gather(*[
            _one_user(client, i, pick_questions(pool, per_user, repeat_ratio, rng), scope, results)
            for i in range(users)
        ])
    wall_s = time.perf_counter() - t0
    ok = [r for r in results if r["status"] == 200]
    lat = [r["ms"] for r in ok]
    statuses: dict[str, int] = {}
    for r in results:
        statuses[str(r["status"])] = statuses.get(str(r["status"]), 0) + 1
    actions: dict[str, int] = {}
    for r in ok:
        actions[r["action"]] = actions.get(r["action"], 0) + 1
    gen_tokens = sum(r["completion_tokens"] for r in ok if not r.get("cached"))
    retry_after = [int(r["retry_after"]) for r in results if r.get("retry_after")]
    stage = {}
    for k in ("retrieval", "generation", "verification"):
        xs = [r["timings_ms"].get(k) for r in ok if r.get("timings_ms", {}).get(k) is not None]
        if xs:
            stage[k] = {"p50": percentile(xs, 50), "p95": percentile(xs, 95)}
    return {
        "users": users, "requests": len(results), "wall_s": round(wall_s, 2),
        "answers_per_min": round(len(ok) / wall_s * 60, 1) if wall_s else None,
        "latency_ms": {"p50": percentile(lat, 50), "p95": percentile(lat, 95),
                       "p99": percentile(lat, 99), "max": percentile(lat, 100)},
        "stage_ms": stage,
        "status": statuses, "actions": actions,
        "cache_hit_share": (round(sum(1 for r in ok if r.get("cached")) / len(ok), 3)
                            if ok else None),
        "generator_tokens_per_s": round(gen_tokens / wall_s, 1) if wall_s else None,
        "retry_after_s": ({"min": min(retry_after), "max": max(retry_after)}
                          if retry_after else None),
        "errors": sorted({r["error"] for r in results if r.get("error")}),
    }  # fmt: skip


def write_report(summary: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = [f"# Load test — {summary['base_url']} — {summary['started_at']}", "",
             f"profile: {summary.get('profile', '?')} · state backend: "
             f"{summary.get('state_backend', '?')} · repeat ratio {summary['repeat_ratio']} · "
             f"{summary['requests_per_user']} requests per user", "",
             "| users | answers/min | p50 ms | p95 ms | p99 ms | 200 | 429 | 503 | other | "
             "cache hits | gen tok/s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lv in summary["levels"]:
        st = lv["status"]
        other = sum(v for k, v in st.items() if k not in ("200", "429", "503"))
        la = lv["latency_ms"]
        lines.append(f"| {lv['users']} | {lv['answers_per_min']} | {la['p50']} | {la['p95']} | "
                     f"{la['p99']} | {st.get('200', 0)} | {st.get('429', 0)} | "
                     f"{st.get('503', 0)} | {other} | {lv['cache_hit_share']} | "
                     f"{lv['generator_tokens_per_s']} |")
    lines += ["", "Stage latencies (p50 / p95 ms) at the highest level:", ""]
    top = summary["levels"][-1]["stage_ms"]
    for k, v in top.items():
        lines.append(f"- {k}: {v['p50']} / {v['p95']}")
    lines += ["", "A 503 is admission control refusing a request the queue could not hold; its "
              "Retry-After tells the client when to retry. Numbers from test doubles measure "
              "service overhead only."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def run(make_client, *, base_url: str, users: list[int], per_user: int, pool: list[str],
        repeat_ratio: float, scope: str | None, out: Path, seed: int = 7,
        health: dict | None = None) -> dict[str, Any]:  # fmt: skip
    summary: dict[str, Any] = {
        "base_url": base_url, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "requests_per_user": per_user, "repeat_ratio": repeat_ratio, "scope": scope,
        "profile": (health or {}).get("profile"),
        "state_backend": (health or {}).get("state_backend"),
        "concurrency": (health or {}).get("concurrency"), "levels": [],
    }  # fmt: skip
    for i, n in enumerate(users):
        lvl = asyncio.run(run_level(make_client, n, per_user, pool, repeat_ratio, scope, seed + i))
        summary["levels"].append(lvl)
        print(json.dumps({k: lvl[k] for k in ("users", "answers_per_min", "latency_ms", "status")}),
              flush=True)
    write_report(summary, out)
    return summary


def main(argv: list[str] | None = None) -> int:
    import httpx

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8471")
    ap.add_argument("--token", default="", help="bearer token (or set AEOS_TOKEN)")
    ap.add_argument("--api-key", default="", help="X-API-Key instead of a token")
    ap.add_argument("--users", type=int, nargs="+", default=[10, 50, 100])
    ap.add_argument("--requests-per-user", type=int, default=5)
    ap.add_argument("--questions", type=Path, default=None, help="jsonl ({question}) or txt")
    ap.add_argument("--repeat-ratio", type=float, default=0.5)
    ap.add_argument("--scope", default=None, choices=[None, "indexed", "uploaded", "both"])
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    import os

    headers = {}
    tok = a.token or os.environ.get("AEOS_TOKEN", "")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    if a.api_key:
        headers["X-API-Key"] = a.api_key
    health = httpx.get(a.base_url.rstrip("/") + "/health", timeout=30).json()

    def make_client():
        return httpx.AsyncClient(base_url=a.base_url, headers=headers, timeout=a.timeout)

    run(make_client, base_url=a.base_url, users=a.users, per_user=a.requests_per_user,
        pool=load_questions(a.questions), repeat_ratio=a.repeat_ratio, scope=a.scope, out=a.out,
        health=health)  # fmt: skip
    print(f"report: {a.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
