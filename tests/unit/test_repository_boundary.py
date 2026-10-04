"""The repository boundary document must exist and name the correct source and target."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "REPOSITORY_BOUNDARY.md"


def test_boundary_doc_exists():
    assert DOC.is_file(), "docs/REPOSITORY_BOUNDARY.md is required"


def test_boundary_doc_names_source_and_target():
    text = DOC.read_text(encoding="utf-8")
    assert "samehelmalt/LitBioRAG\nREAD ONLY" in text
    assert "samehelmalt/LitBioRAG-v2-Production\nREAD/WRITE" in text


def test_no_thesis_drive_paths_in_tree():
    """Colab/Drive paths from the thesis must not leak into production code."""
    offenders = []
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts or path == Path(__file__):
            continue
        if "/content/drive/" in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"hard-coded Drive paths found: {offenders}"
