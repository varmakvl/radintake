"""Local OCR tool: words, boxes, confidences and page orientation. CPU only, fully offline.

Engine: Tesseract 5 (``configs/models.yaml: models.ocr``) through ``pytesseract``. Chosen because
it runs on CPU in CI (``apt install tesseract-ocr``), needs no GPU memory (the 4 GB RTX 3050 is
left entirely to the VLM), makes no network call, and is deterministic for identical input
(single-threaded: ``OMP_THREAD_LIMIT=1``). Measured latency: see ``docs/ocr-engine.md``.

Coordinates: every box is in the pixel grid of the image passed in (``[x0, y0, x1, y1]``).
Map to original page coordinates with ``Page.to_original`` - this tool never rescales.

``orientation(img)`` returns the clockwise rotation (0/90/180/270) that makes the page upright.

Engine switch: ``ocr()`` dispatches on the active engine - ``set_engine()`` (the pipeline calls it with
``Config.ocr_engine``), else ``RADINTAKE_OCR_ENGINE``, else ``ocr.engine`` from the config files;
default ``tesseract``. ``paddle`` runs the optional PaddleOCR backend (``radintake.tools.ocr_paddle``)
with the same ``OcrWord`` contract and raises ``OcrUnavailable`` naming the extra when it is not
installed. ``orientation()`` is always Tesseract.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from radintake.core.types import BBox

OCR_TOOL_NAME = "tesseract"
_WIN_DEFAULTS = (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                 r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # determinism: single-threaded Tesseract

_configured = False
_version: str | None = None
_engine_override: str | None = None


class OcrUnavailable(RuntimeError):
    """The selected OCR engine cannot run (Tesseract binary or the PaddleOCR extra missing)."""


def set_engine(name: str | None) -> None:
    """Pin the OCR engine for this process (``None`` -> back to env/config resolution)."""
    global _engine_override
    if name is not None:
        from radintake.config import OCR_ENGINES

        name = name.strip().lower()
        if name not in OCR_ENGINES:
            raise ValueError(f"unknown OCR engine {name!r}; expected one of {', '.join(OCR_ENGINES)}")
    _engine_override = name


def active_engine() -> str:
    """Engine ``ocr()`` will use: set_engine() > RADINTAKE_OCR_ENGINE > ocr.engine > tesseract."""
    if _engine_override is not None:
        return _engine_override
    from radintake.config import load_config

    return load_config().ocr_engine


@dataclass(frozen=True)
class OcrWord:
    text: str
    bbox: BBox
    conf: float  # 0..1
    block: int
    par: int
    line: int
    word: int

    @property
    def line_id(self) -> tuple[int, int, int]:
        return (self.block, self.par, self.line)

    def to_word(self, page: int = 1):
        from radintake.core.types import Word

        return Word(text=self.text, bbox=self.bbox, confidence=self.conf, page=page)


def _tesseract_cmd() -> str:
    env = os.environ.get("TESSERACT_CMD")
    if env:
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    for p in _WIN_DEFAULTS:
        if Path(p).is_file():
            return p
    raise OcrUnavailable("tesseract binary not found (apt install tesseract-ocr)")


def _tess():
    global _configured
    import pytesseract

    if not _configured:
        pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd()
        _configured = True
    return pytesseract


def tool_version() -> str:
    """Engine name+version string recorded in traces, e.g. ``tesseract 5.4.0``."""
    global _version
    if _version is None:
        out = subprocess.run([_tesseract_cmd(), "--version"], capture_output=True, text=True,
                             timeout=30, check=False)
        m = re.search(r"tesseract\s+v?([0-9][^\s]*)", out.stdout + out.stderr)
        _version = f"{OCR_TOOL_NAME} {m.group(1) if m else 'unknown'}"
    return _version


def _prep(img: Any) -> np.ndarray:
    from PIL import Image

    if isinstance(img, Image.Image):
        img = np.asarray(img.convert("L") if img.mode not in ("L", "RGB") else img)
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    return np.ascontiguousarray(a)


def ocr(page_image: Any, *, lang: str = "eng", psm: int = 6, trace: Any = None) -> list[OcrWord]:
    """Words with boxes (input-image coords) and confidences in 0..1.

    ``psm`` 6 (uniform block) suits report pages; the words are returned in reading order.
    When ``trace`` (a ``CaseTrace``) is given, a ``tool_call`` event records the tool name and
    version plus counts only (never text).
    """
    from PIL import Image

    if active_engine() == "paddle":
        from radintake.tools import ocr_paddle

        return ocr_paddle.ocr(page_image, lang=lang, trace=trace)
    pt = _tess()
    t0 = time.perf_counter()
    arr = _prep(page_image)
    d = pt.image_to_data(Image.fromarray(arr), lang=lang, config=f"--psm {psm} --oem 1",
                         output_type=pt.Output.DICT, timeout=120)
    words: list[OcrWord] = []
    for i, text in enumerate(d["text"]):
        text = (text or "").strip()
        conf = float(d["conf"][i])
        if not text or conf < 0:
            continue
        x, y, w, h = (int(d[k][i]) for k in ("left", "top", "width", "height"))
        if w <= 0 or h <= 0:
            continue
        words.append(OcrWord(text=text, bbox=BBox(x, y, x + w, y + h), conf=min(max(conf / 100.0, 0.0), 1.0),
                             block=int(d["block_num"][i]), par=int(d["par_num"][i]),
                             line=int(d["line_num"][i]), word=int(d["word_num"][i])))
    if trace is not None:
        trace.tool_call("ocr", {"lang": lang, "psm": psm, "w": int(arr.shape[1]), "h": int(arr.shape[0])},
                        {"n_words": len(words),
                         "mean_conf": round(float(np.mean([w.conf for w in words])) if words else 0.0, 4)},
                        round((time.perf_counter() - t0) * 1000.0, 3), tool_version=tool_version())
    return words


def _rot_cw(a: np.ndarray, deg: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(a, k=(-deg // 90) % 4))


def _score(a: np.ndarray, pt) -> float:
    from PIL import Image

    d = pt.image_to_data(Image.fromarray(a), config="--psm 6 --oem 1", output_type=pt.Output.DICT, timeout=60)
    good = [float(c) for t, c in zip(d["text"], d["conf"], strict=True)
            if (t or "").strip() and float(c) >= 0 and re.search(r"[A-Za-z]{2,}", t)]
    return float(sum(good)) / 100.0


def orientation(page_image: Any, *, trace: Any = None) -> int:
    """Clockwise rotation in {0, 90, 180, 270} that makes the page upright.

    Tesseract OSD decides when it is confident; otherwise (blurry photos, sparse pages) the four
    rotations of a downscaled copy are scored by summed word confidence and the best wins.
    """
    pt = _tess()
    t0 = time.perf_counter()
    a = _prep(page_image)
    h, w = a.shape[:2]
    scale = 1600.0 / max(h, w)
    small = a
    if scale < 1.0:
        import cv2

        small = cv2.resize(a, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    method, angle, osd_conf = "osd", None, 0.0
    try:
        from PIL import Image

        osd = pt.image_to_osd(Image.fromarray(small), output_type=pt.Output.DICT, timeout=60)
        angle, osd_conf = int(osd["rotate"]) % 360, float(osd.get("orientation_conf", 0.0))
    except Exception:  # noqa: BLE001 - OSD fails on sparse/blank pages; fall back to scoring
        angle = None
    if angle not in (0, 90, 180, 270) or osd_conf < 2.0:
        method = "rotation_scoring"
        scores = {deg: _score(_rot_cw(small, deg), pt) for deg in (0, 90, 180, 270)}
        angle = max(scores, key=lambda k: (scores[k], -k))
    if trace is not None:
        trace.tool_call("ocr_orientation", {}, {"rotate_cw": angle, "method": method,
                                                "osd_conf": round(osd_conf, 3)},
                        round((time.perf_counter() - t0) * 1000.0, 3), tool_version=tool_version())
    return int(angle)
