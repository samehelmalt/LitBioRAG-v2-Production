"""Native chart data reaches the slide text; client-mode guard; token issuer."""

import pytest

pytest.importorskip("pptx")

from pathlib import Path  # noqa: E402

from scripts import entrypoint, issue_client_token  # noqa: E402
from scripts.corpus.pptx_decks import extract_deck  # noqa: E402


def test_chart_series_are_extracted_from_the_chart_object(tmp_path):
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE
    from pptx.util import Inches

    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "LDL reduction by statin"
    data = CategoryChartData()
    data.categories = ["atorvastatin", "simvastatin"]
    data.add_series("LDL reduction %", (39, 27))
    s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1.5), Inches(6),
                       Inches(4), data)  # fmt: skip
    p = tmp_path / "chart.pptx"
    prs.save(str(p))
    docs = extract_deck(p, course_id="c", lecture_id="l", lecture_title="t")
    text = docs[0].abstract
    assert "Chart" in text and "LDL reduction % — atorvastatin: 39; simvastatin: 27" in text
    assert "data from the chart object" in text  # not a pixel estimate


def test_client_mode_guard_refuses_demo_setups(monkeypatch):
    monkeypatch.setenv("LITBIORAG_JWT_SECRET", "x" * 40)
    with pytest.raises(SystemExit):
        entrypoint._client_mode_guard(Path("data/artifacts/demo"), "test")
    with pytest.raises(SystemExit):
        entrypoint._client_mode_guard(Path("data/artifacts/real"), "test")
    monkeypatch.setenv("LITBIORAG_JWT_SECRET", "short")
    with pytest.raises(SystemExit):
        entrypoint._client_mode_guard(Path("data/artifacts/real"), "hosted")
    monkeypatch.setenv("LITBIORAG_JWT_SECRET", "x" * 40)
    monkeypatch.setenv("AEOS_DEMO_TOKENS", "1")
    with pytest.raises(SystemExit):  # self-service demo tokens never in a client trial
        entrypoint._client_mode_guard(Path("data/artifacts/real"), "hosted")
    monkeypatch.delenv("AEOS_DEMO_TOKENS")
    entrypoint._client_mode_guard(Path("data/artifacts/real"), "hosted")  # passes


def test_issue_client_token(tmp_path, monkeypatch):
    import jwt

    secret = "s" * 40
    tok = issue_client_token.issue(secret, subject="prospect-001", days=7, courses=["IM-2026"],
                                   roles=[], tenant="clinic-a")  # fmt: skip
    claims = jwt.decode(tok, secret, algorithms=["HS256"], audience="litbiorag")
    assert claims["sub"] == "prospect-001" and claims["tenant"] == "clinic-a"
    for bad in ({"subject": "demo-admin"}, {"days": 90}):
        with pytest.raises(ValueError):
            issue_client_token.issue(secret, **{"subject": "p", "days": 7, "courses": [],
                                                "roles": [], **bad})
    with pytest.raises(ValueError):
        issue_client_token.issue("short", subject="p", days=7, courses=[], roles=[])
    env = tmp_path / ".env"
    env.write_text("LITBIORAG_JWT_SECRET=\"" + secret + "\"  # comment\nOTHER=1\n")
    assert issue_client_token.read_env_file(env)["LITBIORAG_JWT_SECRET"] == secret
