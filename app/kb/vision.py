"""Text from slide and page images: OCR (Tesseract) and/or a hosted vision model, both optional.

A medically important slide may be an ECG, a histology image or a flowchart with no text in the
PowerPoint shapes. Without this, such slides contribute nothing searchable. Extracted text is
appended to the unit as ``[image N text] …`` or ``[image N description] …`` so the verifier and
the citation still point at the slide; units whose images produced no text are flagged
``needs_ocr`` so coverage gaps are visible instead of silent (KB status, Knowledge Base page).

Configuration: ``configs/kb.yaml: images``. Nothing here is required: with ``ocr: none`` and
``captions: none`` the extractor is a no-op and only the flag is produced.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import yaml

log = logging.getLogger("litbiorag.kb.vision")
_ROOT = Path(__file__).resolve().parents[2]


class ImageTextExtractor(Protocol):
    name: str

    def extract(self, image_bytes: bytes, *, content_type: str = "image/png",
                context: str = "") -> str: ...  # fmt: skip


class NoopImageText:
    name = "none"

    def extract(self, image_bytes: bytes, *, content_type: str = "image/png",
                context: str = "") -> str:  # fmt: skip
        return ""


class TesseractOCR:
    """Printed text in images via pytesseract (needs the tesseract binary)."""

    name = "tesseract"

    def __init__(self, lang: str = "eng"):
        import pytesseract  # optional

        if shutil.which("tesseract") is None:
            raise RuntimeError("tesseract binary not found on PATH")
        self._t = pytesseract
        self.lang = lang

    def extract(self, image_bytes: bytes, *, content_type: str = "image/png",
                context: str = "") -> str:  # fmt: skip
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return (self._t.image_to_string(img, lang=self.lang) or "").strip()


class HostedVisionCaptioner:
    """Describe an image with an OpenAI-compatible vision model (key from the environment)."""

    name = "hosted-vision"
    PROMPT = ("You are transcribing a medical teaching slide image. Transcribe all visible text "
              "exactly, then describe the medical content of charts, diagrams, images and "
              "flowcharts in neutral terms (what is shown, labels, axes, values). Do not "
              "diagnose, do not add information that is not visible. Plain text only.")

    def __init__(self, model_id: str, *, endpoint: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", timeout: int = 90,
                 max_tokens: int = 600):  # fmt: skip
        self.model_id = model_id
        self.endpoint = endpoint.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.name = f"hosted-vision:{model_id}"
        if not self.api_key:
            raise RuntimeError(f"{api_key_env} is not set")

    def extract(self, image_bytes: bytes, *, content_type: str = "image/png",
                context: str = "") -> str:  # fmt: skip
        data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode()}"
        body = {
            "model": self.model_id, "max_tokens": self.max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": self.PROMPT + (f"\nSlide context: {context}"
                                                        if context else "")},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
        }  # fmt: skip
        req = urllib.request.Request(
            f"{self.endpoint}/chat/completions", json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        return (data["choices"][0]["message"]["content"] or "").strip()


class CompositeImageText:
    """OCR first; ask the captioner only when OCR yields less than ``min_ocr_chars``."""

    def __init__(self, ocr: ImageTextExtractor | None, captioner: ImageTextExtractor | None,
                 *, min_ocr_chars: int = 40):  # fmt: skip
        self.ocr, self.captioner, self.min_ocr_chars = ocr, captioner, min_ocr_chars
        self.name = "+".join(x.name for x in (ocr, captioner) if x) or "none"

    def extract(self, image_bytes: bytes, *, content_type: str = "image/png",
                context: str = "") -> str:  # fmt: skip
        text = ""
        if self.ocr is not None:
            try:
                text = self.ocr.extract(image_bytes, content_type=content_type, context=context)
            except Exception as e:  # noqa: BLE001
                log.warning("ocr failed: %s", e)
        if len(text) < self.min_ocr_chars and self.captioner is not None:
            try:
                cap = self.captioner.extract(image_bytes, content_type=content_type,
                                             context=context)  # fmt: skip
                text = f"{text}\n{cap}".strip() if text else cap
            except Exception as e:  # noqa: BLE001
                log.warning("vision caption failed: %s", e)
        return text


def load_image_config(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path or os.environ.get("LITBIORAG_KB_CONFIG") or _ROOT / "configs" / "kb.yaml")
    raw = yaml.safe_load(p.read_text()) if p.exists() else {}
    return (raw or {}).get("images") or {}


def make_image_extractor(cfg: dict[str, Any] | None = None) -> ImageTextExtractor:
    """Build from config; a missing engine degrades to a no-op with a logged reason."""
    c = cfg if cfg is not None else load_image_config()
    ocr = cap = None
    if c.get("ocr", "none") == "tesseract":
        try:
            ocr = TesseractOCR(lang=str(c.get("ocr_lang", "eng")))
        except Exception as e:  # noqa: BLE001
            log.warning("OCR disabled: %s", e)
    if c.get("captions", "none") == "hosted":
        try:
            cap = HostedVisionCaptioner(
                str(c.get("model", "gpt-4o-mini")), endpoint=str(c.get("endpoint",
                                                                       "https://api.openai.com/v1")),
                api_key_env=str(c.get("api_key_env", "OPENAI_API_KEY")),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("vision captions disabled: %s", e)
    if ocr is None and cap is None:
        return NoopImageText()
    return CompositeImageText(ocr, cap, min_ocr_chars=int(c.get("min_ocr_chars", 40)))
