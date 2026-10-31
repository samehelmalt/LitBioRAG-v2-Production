import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jwt")

from fastapi.testclient import TestClient  # noqa: E402

from app.api.main import create_app  # noqa: E402
from app.api.settings import ApiSettings, AuditSettings, AuthSettings, LimitSettings  # noqa: E402
from app.core.budget import UsageLedger  # noqa: E402
from app.service.pipeline import AnswerPipeline  # noqa: E402
from tests.retrieval.conftest import fixture_bundle  # noqa: E402, F401

SECRET = "test-secret-not-for-production-0123456789abcdef"


@pytest.fixture(scope="session")
def pipeline(fixture_bundle):  # noqa: F811
    return AnswerPipeline(fixture_bundle, profile="test", ledger=UsageLedger(None))


def make_client(pipeline, tmp_path, **auth) -> TestClient:
    st = ApiSettings(
        bundle=str(pipeline.bundle), profile="test",
        auth=AuthSettings(**{"required": False, "anonymous_courses": [], **auth}),
        limits=LimitSettings(rate_per_minute=600, burst=1000),
        audit=AuditSettings(path=str(tmp_path / "audit.jsonl")),
    )  # fmt: skip
    return TestClient(create_app(st, pipeline=pipeline, ledger=UsageLedger(None)))


@pytest.fixture
def client(pipeline, tmp_path, monkeypatch):
    monkeypatch.setenv("LITBIORAG_JWT_SECRET", SECRET)
    with make_client(pipeline, tmp_path) as c:
        c.audit_path = tmp_path / "audit.jsonl"
        yield c


def token(sub="student-1", courses=(), roles=(), secret=SECRET, aud="litbiorag"):
    import time

    import jwt

    return jwt.encode({"sub": sub, "courses": list(courses), "roles": list(roles), "aud": aud,
                       "exp": int(time.time()) + 600}, secret, algorithm="HS256")  # fmt: skip
