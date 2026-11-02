"""Shared state for several API replicas: Redis variants of the per-process stores.

One API process keeps its rate limiter in memory, the trial and usage ledgers in SQLite / JSONL
and the knowledge-base job table in the KB registry's SQLite. Two replicas behind a load balancer
would each enforce their own limits and caps, and a job started on one would be invisible on the
other. ``configs/api.yaml: state_backend: redis`` (plus ``REDIS_URL``) swaps in the classes below,
which keep the same interfaces, so nothing else in the service changes:

* ``RedisRateLimiter.allow(key)``          token bucket, atomic Lua script
* ``RedisTrialLedger.reserve/refund/used`` request-id idempotency and per-subject count, atomic
* ``RedisUsageLedger``                     today's spend as a shared counter (JSONL still appended)
* ``RedisJobTable``                        KB index jobs as hashes, same dict-like interface

Not moved here: answer records, chats, review cases and KB files/indexes. Those stay on one node
(SQLite + local disk) or move to PostgreSQL and shared storage; see docs/DEPLOYMENT.md "Scaling".
The in-process admission queue is per replica on purpose (each replica owns its model slots).
``redis`` is an optional dependency (``scale`` extra); tests use ``fakeredis``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.api.trials import TrialLimitExceeded
from app.core.budget import Pricing, Usage, UsageLedger

# ---------------------------------------------------------------------------- rate limiter ------
_BUCKET_LUA = """
local key, rate, burst, now = KEYS[1], tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
local b = redis.call('HMGET', key, 'tokens', 'last')
local tokens = tonumber(b[1]); local last = tonumber(b[2])
if tokens == nil then tokens = burst; last = now end
tokens = math.min(burst, tokens + (now - last) * rate)
local allowed = 0; local retry = 0
if tokens >= 1 then tokens = tokens - 1; allowed = 1
elseif rate > 0 then retry = (1 - tokens) / rate else retry = 60 end
redis.call('HSET', key, 'tokens', tokens, 'last', now)
redis.call('EXPIRE', key, math.ceil(burst / math.max(rate, 0.001)) + 60)
return {allowed, tostring(retry)}
"""


class RedisRateLimiter:
    def __init__(self, client, rate_per_minute: int, burst: int, *, prefix: str = "aeos:rl:"):
        self.r = client
        self.rate = rate_per_minute / 60.0
        self.burst = max(1, burst)
        self.prefix = prefix
        self._script = client.register_script(_BUCKET_LUA)

    def allow(self, key: str, now: float | None = None) -> tuple[bool, float]:
        now = time.time() if now is None else now
        allowed, retry = self._script(keys=[self.prefix + key], args=[self.rate, self.burst, now])
        return bool(int(allowed)), float(retry)


# ---------------------------------------------------------------------------- trial ledger ------
_RESERVE_LUA = """
local req, subj, limit = KEYS[1], KEYS[2], tonumber(ARGV[1])
local owner = redis.call('GET', req)
if owner then
  if owner ~= ARGV[2] then return {-2, 0} end
  return {1, redis.call('SCARD', subj)}
end
local n = redis.call('SCARD', subj)
if n >= limit then return {-1, n} end
redis.call('SET', req, ARGV[2])
redis.call('SADD', subj, ARGV[3])
return {1, n + 1}
"""


class RedisTrialLedger:
    """Same contract as ``app.api.trials.TrialLedger``; keys never expire (a trial is a trial)."""

    def __init__(self, client, limit: int = 5, *, prefix: str = "aeos:trial:"):
        if limit < 1:
            raise ValueError("limit must be positive")
        self.r = client
        self.limit = limit
        self.prefix = prefix
        self._reserve = client.register_script(_RESERVE_LUA)

    def _k(self, kind: str, value: str) -> str:
        return f"{self.prefix}{kind}:{value}"

    def used(self, subject: str) -> int:
        return int(self.r.scard(self._k("subject", subject)))

    def remaining(self, subject: str) -> int:
        return max(0, self.limit - self.used(subject))

    def reserve(self, subject: str, request_id: str) -> int:
        if not subject or subject == "anonymous":
            raise TrialLimitExceeded("sign in to use the trial")
        code, n = self._reserve(keys=[self._k("request", request_id), self._k("subject", subject)],
                                args=[self.limit, subject, request_id])  # fmt: skip
        code = int(code)
        if code == -2:
            raise TrialLimitExceeded("request id belongs to another identity")
        if code == -1:
            raise TrialLimitExceeded(f"trial allowance of {self.limit} questions is used up")
        return max(0, self.limit - int(n))

    def refund(self, subject: str, request_id: str) -> bool:
        owner = self.r.get(self._k("request", request_id))
        if owner is None or (owner.decode() if isinstance(owner, bytes) else owner) != subject:
            return False
        pipe = self.r.pipeline()
        pipe.delete(self._k("request", request_id))
        pipe.srem(self._k("subject", subject), request_id)
        return bool(pipe.execute()[1])


# ---------------------------------------------------------------------------- usage ledger ------
class RedisUsageLedger(UsageLedger):
    """Today's spend is a shared Redis counter, so the daily cap holds across replicas. Each
    replica still appends its own JSONL lines (the audit of what was spent where)."""

    def __init__(self, client, path: Path | str | None, *, max_usd_per_request: float = 0.0,
                 max_usd_per_day: float = 0.0, pricing: Pricing | None = None,
                 prefix: str = "aeos:usage:"):  # fmt: skip
        self.r = client
        self.prefix = prefix
        super().__init__(path, max_usd_per_request=max_usd_per_request,
                         max_usd_per_day=max_usd_per_day, pricing=pricing)  # fmt: skip

    def _key(self) -> str:
        return self.prefix + time.strftime("%Y-%m-%d")

    def _replay_today(self) -> float:
        v = self.r.get(self._key())
        return float(v) if v is not None else 0.0

    @property
    def spent_today(self) -> float:
        return self._replay_today()

    def check_affordable(self, estimated_usd: float) -> None:
        self._spent_today = self._replay_today()
        super().check_affordable(estimated_usd)

    def record(self, usage: Usage, *, request_id: str = "", role: str = "generator") -> Usage:
        self.pricing.price(usage)
        key = self._key()
        pipe = self.r.pipeline()
        pipe.incrbyfloat(key, usage.cost_usd)
        pipe.expire(key, 3 * 86400)
        pipe.execute()
        if self.path:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "request_id": request_id,
                       "role": role, **usage.to_dict()}  # fmt: skip
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
        return usage


# ---------------------------------------------------------------------------- KB job table ------
_JOB_FIELDS = ("job_id", "kb_id", "file_id", "status", "started_at", "finished_at", "n_units",
               "needs_ocr_units", "error")  # fmt: skip


class RedisJobTable:
    """Drop-in for ``app.kb.registry.JobTable``: one hash per job, an ordered set for listing."""

    def __init__(self, client, *, prefix: str = "aeos:kbjob:", ttl_s: int = 30 * 86400):
        self.r = client
        self.prefix = prefix
        self.ttl_s = ttl_s

    def _k(self, job_id: str) -> str:
        return self.prefix + job_id

    def start(self, job_id: str, *, kb_id: str, file_id: str) -> dict[str, Any]:
        now = time.time()
        pipe = self.r.pipeline()
        pipe.delete(self._k(job_id))
        pipe.hset(self._k(job_id), mapping={"job_id": job_id, "kb_id": kb_id, "file_id": file_id,
                                            "status": "queued", "started_at": now})  # fmt: skip
        pipe.expire(self._k(job_id), self.ttl_s)
        pipe.zadd(self.prefix + "_all", {job_id: now})
        pipe.execute()
        return self[job_id]

    def update(self, job_id: str, **fields: Any) -> None:
        upd = {k: ("" if v is None else v) for k, v in fields.items()
               if k in ("status", "finished_at", "n_units", "needs_ocr_units", "error")}
        if upd:
            self.r.hset(self._k(job_id), mapping=upd)

    def _decode(self, raw: dict) -> dict[str, Any]:
        def s(x):
            return x.decode() if isinstance(x, bytes) else x

        d = {s(k): s(v) for k, v in raw.items()}
        out: dict[str, Any] = {f: None for f in _JOB_FIELDS}
        for k, v in d.items():
            if v == "":
                out[k] = None
            elif k in ("started_at", "finished_at"):
                out[k] = float(v)
            elif k in ("n_units", "needs_ocr_units"):
                out[k] = int(float(v))
            else:
                out[k] = v
        return out

    def __getitem__(self, job_id: str) -> dict[str, Any]:
        raw = self.r.hgetall(self._k(job_id))
        if not raw:
            raise KeyError(job_id)
        return self._decode(raw)

    def get(self, job_id: str, default=None):
        try:
            return self[job_id]
        except KeyError:
            return default

    def items(self):
        ids = self.r.zrange(self.prefix + "_all", 0, -1)
        out = []
        for jid in ids:
            jid = jid.decode() if isinstance(jid, bytes) else jid
            j = self.get(jid)
            if j is None:
                self.r.zrem(self.prefix + "_all", jid)  # expired
                continue
            out.append((jid, j))
        return out

    def values(self):
        return [v for _, v in self.items()]


# ---------------------------------------------------------------------------- selection ---------
def connect(url: str):
    import redis  # optional dependency: pip install ".[scale]"

    client = redis.Redis.from_url(url)
    client.ping()
    return client


class StateBackend:
    """Factory used by ``create_app``: ``sqlite`` (default, per process) or ``redis`` (shared)."""

    def __init__(self, kind: str = "sqlite", client=None):
        if kind not in ("sqlite", "redis"):
            raise ValueError("state_backend must be 'sqlite' or 'redis'")
        if kind == "redis" and client is None:
            raise ValueError("redis backend needs a client (REDIS_URL)")
        self.kind = kind
        self.client = client

    @classmethod
    def from_settings(cls, kind: str, url: str) -> StateBackend:
        if kind == "redis":
            if not url:
                raise RuntimeError("state_backend is redis but REDIS_URL is not set")
            return cls("redis", connect(url))
        return cls("sqlite")

    @property
    def shared(self) -> bool:
        return self.kind == "redis"

    def rate_limiter(self, rate_per_minute: int, burst: int):
        if self.shared:
            return RedisRateLimiter(self.client, rate_per_minute, burst)
        from app.api.ratelimit import RateLimiter

        return RateLimiter(rate_per_minute, burst)

    def trial_ledger(self, path: Path | str | None, limit: int):
        if self.shared:
            return RedisTrialLedger(self.client, limit)
        from app.api.trials import TrialLedger

        return TrialLedger(path, limit)

    def usage_ledger(self, path: Path | str | None, **caps) -> UsageLedger:
        if self.shared:
            return RedisUsageLedger(self.client, path, **caps)
        return UsageLedger(path, **caps)

    def kb_jobs(self):
        """None → the KB registry keeps its SQLite job table."""
        return RedisJobTable(self.client) if self.shared else None
