"""Admission control: capacity, FIFO queue with positions, bounded queue, timeout, metrics, and
the API's 503 + Retry-After when the generator queue is full."""

import threading
import time

import pytest

from app.core.admission import Admission, Overloaded, RoleLimits

pytest.importorskip("fastapi")
from tests.api.conftest import fixture_bundle, pipeline  # noqa: E402, F401, F811  (fixtures)


def test_capacity_and_fifo_positions():
    adm = Admission({"generator": RoleLimits(capacity=1, queue_max=4, queue_timeout_s=5)})
    positions: list[int] = []
    order: list[str] = []
    gate = threading.Event()

    def holder():
        with adm.admit("generator"):
            order.append("first")
            gate.wait(2)

    def waiter(name):
        with adm.admit("generator", on_queued=positions.append):
            order.append(name)

    t0 = threading.Thread(target=holder)
    t0.start()
    time.sleep(0.05)
    assert adm.snapshot()["generator"]["in_flight"] == 1
    ts = []
    for n in ("second", "third"):
        t = threading.Thread(target=waiter, args=(n,))
        t.start()
        time.sleep(0.03)  # deterministic queue order
        ts.append(t)
    assert adm.snapshot()["generator"]["queued"] == 2
    assert positions == [0, 1]  # number of callers ahead when queued
    gate.set()
    for t in (t0, *ts):
        t.join(3)
    assert order == ["first", "second", "third"]
    s = adm.snapshot()["generator"]
    assert s["in_flight"] == 0 and s["queued"] == 0 and s["admitted"] == 3 and s["rejected"] == 0


def test_full_queue_is_refused_and_timeout_counts():
    adm = Admission({"generator": RoleLimits(capacity=1, queue_max=1, queue_timeout_s=0.1)})
    release = threading.Event()
    held = threading.Event()

    def hold():
        with adm.admit("generator"):
            held.set()
            release.wait(3)

    t = threading.Thread(target=hold)
    t.start()
    assert held.wait(2)
    # one waiter fits in the queue; it times out after 0.1 s
    with pytest.raises(Overloaded) as to:
        with adm.admit("generator"):
            pass
    assert to.value.reason.startswith("waited too long") and to.value.retry_after_s >= 5
    # while a waiter is queued the next caller is refused immediately
    outcomes: list[str] = []

    def wait_and_time_out():
        try:
            with adm.admit("generator"):
                outcomes.append("admitted")
        except Overloaded as e:
            outcomes.append(e.reason)

    waiting = threading.Thread(target=wait_and_time_out)
    waiting.start()
    time.sleep(0.03)
    assert adm.snapshot()["generator"]["queued"] == 1
    with pytest.raises(Overloaded) as full:
        with adm.admit("generator"):
            pass
    assert "already waiting" in full.value.reason
    waiting.join(2)
    assert outcomes == ["waited too long for a free slot"]
    s = adm.snapshot()["generator"]
    assert s["rejected"] == 1 and s["timed_out"] == 2 and s["in_flight"] == 1  # holder still in
    release.set()
    t.join(2)
    assert adm.snapshot()["generator"]["in_flight"] == 0


def test_unlisted_roles_are_not_gated_and_config_parsing():
    adm = Admission.from_config({"queue_max": 3, "queue_timeout_s": 7,
                                 "roles": {"generator": {"capacity": 2}, "verifier": {}}})
    assert adm.limits("generator") == RoleLimits(2, 3, 7.0)
    assert adm.limits("verifier") == RoleLimits(1, 3, 7.0)
    assert adm.limits("reranker") is None
    with adm.admit("reranker"), adm.admit("reranker"), adm.admit("reranker"):
        assert "reranker" not in adm.snapshot()
    assert Admission.from_config(None).snapshot() == {}


def test_pool_use_is_admitted():
    from app.core.budget import ResourceConfig
    from models.registry.loader import ModelRegistry
    from models.registry.pool import ModelPool

    pool = ModelPool(ModelRegistry.load(profile="test"), ResourceConfig(
        concurrency={"roles": {"generator": {"capacity": 1, "queue_max": 0}}}))
    seen = []
    with pool.use("generator", on_queued=seen.append):
        with pytest.raises(Overloaded):
            with pool.use("generator"):
                pass
    assert seen == [] and pool.admission.snapshot()["generator"]["rejected"] == 1


def test_api_returns_503_with_retry_after_when_busy(pipeline, tmp_path, monkeypatch):  # noqa: F811
    from tests.api.conftest import SECRET, make_client

    monkeypatch.setenv("LITBIORAG_JWT_SECRET", SECRET)
    saved = pipeline.pool.admission
    pipeline.pool.admission = Admission(
        {"generator": RoleLimits(capacity=1, queue_max=0, queue_timeout_s=1)})
    try:
        with make_client(pipeline, tmp_path) as c:
            with pipeline.pool.admission.admit("generator"):  # someone holds the generator
                r = c.post("/api/v1/query", json={"question": "What does MAP3K3 do?"})
                assert r.status_code == 503
                assert int(r.headers["Retry-After"]) >= 5 and "busy" in r.json()["error"]
                h = c.get("/health").json()["concurrency"]["generator"]
                assert h["in_flight"] == 1 and h["rejected"] == 1
            assert c.post("/api/v1/query", json={"question": "What does MAP3K3 do?"}).status_code \
                == 200
    finally:
        pipeline.pool.admission = saved
