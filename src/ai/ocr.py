"""OCR using RapidOCR (ONNX, ~20MB total). Provides a simple sync API.

The RapidOCR runtime is thread-safe per-instance but model files are large,
so we load ONE recognizer per process and let multiple threads call it.
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass

from PIL import Image
from rapidocr_onnxruntime import RapidOCR

_lock = threading.Lock()
_engine: RapidOCR | None = None


def _get_engine() -> RapidOCR:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = RapidOCR()
    return _engine


@dataclass
class OCRBlock:
    text: str
    confidence: float
    box: list


def ocr_image(image_bytes: bytes, *, strict: bool = False) -> list[OCRBlock]:
    """Run OCR on raw image bytes. Returns list of recognized blocks.

    Default callers receive [] on decode/engine failure. strict=True propagates
    failures so a focused review can distinguish them from an empty image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        import numpy as np
        arr = np.array(img)
    except Exception as e:
        if strict:
            raise
        print(f"[OCR] image decode failed: {type(e).__name__}")
        return []

    engine = _get_engine()
    try:
        result, _elapsed = engine(arr)
    except Exception as e:
        if strict:
            raise
        print(f"[OCR] engine call failed: {type(e).__name__}")
        return []

    if not result:
        return []

    blocks = []
    for item in result:
        if len(item) < 3:
            continue
        box, text, score = item[0], item[1], float(item[2])
        if not text or not text.strip():
            continue
        blocks.append(OCRBlock(text=text.strip(), confidence=score, box=box))
    return blocks


def ocr_image_strict(image_bytes: bytes) -> list[OCRBlock]:
    """Keep failure distinct from a successfully read frame with no text."""
    return ocr_image(image_bytes, strict=True)


def ocr_image_text(image_bytes: bytes) -> str:
    """Convenience: OCR an image and return all recognized text joined by newlines."""
    blocks = ocr_image(image_bytes)
    return "\n".join(b.text for b in blocks)
