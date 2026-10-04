import pytest

pytest.importorskip("tantivy")
pytest.importorskip("faiss")

from scripts import build_indexes  # noqa: E402
from tests.retrieval.fixture_corpus import write_fixture_jsonl  # noqa: E402


@pytest.fixture(scope="session")
def fixture_bundle(tmp_path_factory):
    """A complete artifact bundle built from the 60-document fixture with the hash encoder."""
    root = tmp_path_factory.mktemp("fixture_bundle")
    docs = write_fixture_jsonl(root / "docs.jsonl")
    out = root / "artifact"
    rc = build_indexes.main(
        ["--docs", str(docs), "--out", str(out), "--corpus-id", "fixture_v1",
         "--encoder", "hash:128", "--shard-size", "25", "--batch-size", "8"]  # fmt: skip
    )
    assert rc == 0
    return out
