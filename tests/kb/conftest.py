import pytest

from tests.retrieval.conftest import fixture_bundle  # noqa: F401

pytest.importorskip("fastapi")
pytest.importorskip("pypdf")

from fastapi.testclient import TestClient  # noqa: E402

from app.api.main import create_app  # noqa: E402
from app.api.settings import ApiSettings, AuditSettings, AuthSettings, LimitSettings  # noqa: E402
from app.core.budget import UsageLedger  # noqa: E402
from app.service.pipeline import AnswerPipeline  # noqa: E402
from tests.api.conftest import SECRET  # noqa: E402


@pytest.fixture
def kb_client(fixture_bundle, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("LITBIORAG_JWT_SECRET", SECRET)
    pl = AnswerPipeline(fixture_bundle, profile="test", ledger=UsageLedger(None),
                        kb_root=tmp_path / "kb")  # fmt: skip
    st = ApiSettings(bundle=str(fixture_bundle), profile="test", auth=AuthSettings(required=False),
                     limits=LimitSettings(rate_per_minute=600, burst=1000, max_upload_mb=1),
                     audit=AuditSettings(path=str(tmp_path / "audit.jsonl")))  # fmt: skip
    with TestClient(create_app(st, pipeline=pl, ledger=UsageLedger(None))) as c:
        c.pipeline = pl
        yield c
