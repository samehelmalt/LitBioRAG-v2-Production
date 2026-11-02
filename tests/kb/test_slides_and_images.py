"""Deck fidelity: lecture metadata and canonical slide ids survive upload; image text is
extracted through the configured extractor; image-only slides are flagged; deck versions
supersede without breaking old citations."""

import io

import pytest

pytest.importorskip("pptx")

from PIL import Image, ImageDraw  # noqa: E402

from app.kb.adapters import LectureMeta  # noqa: E402
from app.kb.adapters.pptx import PptxAdapter  # noqa: E402
from app.kb.vision import CompositeImageText, NoopImageText, make_image_extractor  # noqa: E402
from tests.api.conftest import token  # noqa: E402
from tests.kb.fixtures import make_pptx  # noqa: E402

OWNER = {"Authorization": f"Bearer {token(sub='lecturer-1')}"}


class FakeOCR:
    name = "fake-ocr"

    def __init__(self, text="ECG shows peaked T waves consistent with hyperkalemia"):
        self.text, self.calls = text, 0

    def extract(self, image_bytes, *, content_type="image/png", context=""):
        self.calls += 1
        return self.text


def _png() -> bytes:
    img = Image.new("RGB", (240, 80), "white")
    ImageDraw.Draw(img).text((10, 30), "peaked T waves", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_deck_with_image(path, slides, image_slide_title="ECG example"):
    from pptx import Presentation
    from pptx.util import Inches

    make_pptx(path, slides)
    prs = Presentation(str(path))
    s = prs.slides.add_slide(prs.slide_layouts[5])  # title only
    s.shapes.title.text = image_slide_title
    s.shapes.add_picture(io.BytesIO(_png()), Inches(1), Inches(1.5), width=Inches(4))
    prs.save(str(path))
    return path


def test_image_text_is_extracted_and_image_only_slides_are_flagged(tmp_path):
    deck = make_deck_with_image(tmp_path / "cardio.pptx", [("Hyperkalemia", ["Peaked T waves"])])
    ocr = FakeOCR()
    units = list(PptxAdapter(image_text=ocr).extract(deck))
    assert ocr.calls == 1
    img_unit = units[-1]
    assert img_unit.n_images == 1 and img_unit.images_with_text == 1
    assert "[image 1 text] ECG shows peaked T waves" in img_unit.text
    # without an engine the slide is kept but flagged
    units = list(PptxAdapter(image_text=NoopImageText()).extract(deck))
    assert units[-1].n_images == 1 and units[-1].images_with_text == 0


def test_lecture_meta_defaults_and_canonical_ids(tmp_path):
    from app.kb.adapters import units_to_documents

    deck = make_deck_with_image(tmp_path / "Lecture 3 - Potassium.pptx",
                                [("Hyperkalemia", ["Peaked T waves on the ECG"])])  # fmt: skip
    lec = LectureMeta.from_mapping({"course_id": "IM-2026", "version": "2026a"},
                                   file_name=deck.name)  # fmt: skip
    assert lec.lecture_id == "Lecture-3-Potassium" and lec.lecture_title == "Lecture 3 - Potassium"
    docs = units_to_documents(PptxAdapter(image_text=NoopImageText()).extract(deck), kb_id="kb1",
                              file_id="abc", file_name=deck.name, kind="pptx", lecture=lec)
    assert len(docs) == 2  # the image-only slide is kept as a flagged unit
    d0, d1 = docs
    assert d0.source_type.value == "course_slide" and d0.file.course_id == "IM-2026"
    assert d0.course_id == "kb1"  # entitlement key stays the knowledge base
    assert d0.file.canonical_id.startswith("IM-2026/Lecture-3-Potassium/s1/v2026a/")
    assert d1.file.needs_ocr and "image-only slide" in d1.abstract


def test_upload_with_lecture_metadata_cites_slides_and_versions_supersede(kb_client, tmp_path):
    c = kb_client
    c.pipeline.kb.image_text = FakeOCR()
    kb = c.post("/api/v1/kb", json={"name": "Internal medicine"}, headers=OWNER).json()
    kb_id = kb["kb_id"]
    deck1 = make_deck_with_image(tmp_path / "potassium.pptx",
                                 [("Hyperkalemia", ["Hyperkalemia causes peaked T waves and can "
                                                    "progress to asystole."])])  # fmt: skip
    form = {"course_id": "IM-2026", "lecture_id": "L03", "lecture_title": "Potassium disorders",
            "academic_year": "2026", "version": "2026a"}  # fmt: skip
    with open(deck1, "rb") as fh:
        r = c.post(f"/api/v1/kb/{kb_id}/upload", headers=OWNER, data=form,
                   files={"file": (deck1.name, fh, "application/octet-stream")})  # fmt: skip
    assert r.status_code == 202 and r.json()["status"] == "indexed", r.text
    f1 = r.json()["file_id"]
    status = c.get(f"/api/v1/kb/{kb_id}/status", headers=OWNER).json()
    assert status["files"][0]["meta"]["lecture_id"] == "L03"
    assert status["files"][0]["meta"]["n_images"] == 1
    q = c.post("/api/v1/query", json={"question": "What ECG changes does hyperkalemia cause?",
                                      "scope": "uploaded"}, headers=OWNER).json()
    top = q["sources"][0]  # slide 1 (text) or slide 2 (OCR text): both are lecture slides
    assert top["citation_id"].startswith("SLIDE:L03:") and top["tier"] == "course_material"
    assert top["display_source"].startswith("Potassium disorders, Slide ")
    assert top["display_source"].endswith("(v2026a)")
    assert top["canonical_id"].startswith("IM-2026/L03/s") and "/v2026a/" in top["canonical_id"]
    s = c.get(f"/api/v1/slides/{f1}/1", headers=OWNER).json()
    assert s["lecture_title"] == "Potassium disorders" and s["neighbours"]["2"]["title"]
    assert c.get(f"/api/v1/slides/{f1}/2", headers=OWNER).json()["images_with_text"] == 1
    other = {"Authorization": f"Bearer {token(sub='someone')}"}
    assert c.get(f"/api/v1/slides/{f1}/1", headers=other).status_code == 403

    # a new version of the same lecture supersedes the old one in search, not in citations
    deck2 = make_deck_with_image(tmp_path / "potassium_v2.pptx",
                                 [("Hyperkalemia", ["Hyperkalemia causes peaked T waves; treat "
                                                    "with calcium gluconate first."])])  # fmt: skip
    with open(deck2, "rb") as fh:
        r = c.post(f"/api/v1/kb/{kb_id}/upload", headers=OWNER,
                   data={**form, "version": "2027a"},
                   files={"file": (deck2.name, fh, "application/octet-stream")})  # fmt: skip
    f2 = r.json()["file_id"]
    assert f2 != f1
    q = c.post("/api/v1/query", json={"question": "What ECG changes does hyperkalemia cause?",
                                      "scope": "uploaded"}, headers=OWNER).json()
    assert all(s["deck_version"] == "2027a" for s in q["sources"]
               if s["citation_id"].startswith("SLIDE:L03"))  # fmt: skip
    old = c.get(f"/api/v1/slides/{f1}/1", headers=OWNER).json()
    assert old["superseded_by"] == f2 and "peaked T waves" in old["text"]  # still resolvable
    st = c.get(f"/api/v1/kb/{kb_id}/status", headers=OWNER).json()
    by_id = {f["file_id"]: f for f in st["files"]}
    assert by_id[f1]["meta"]["superseded_by"] == f2


def test_make_image_extractor_degrades_gracefully(monkeypatch):
    assert make_image_extractor({"ocr": "none", "captions": "none"}).name == "none"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ext = make_image_extractor({"ocr": "tesseract", "captions": "hosted"})
    assert ext.name == "none"  # neither engine available here; nothing crashes
    comp = CompositeImageText(FakeOCR("short"), FakeOCR("long caption text here"), min_ocr_chars=10)
    assert comp.extract(b"x") == "short\nlong caption text here"
