"""Optional PaddleOCR (PP-OCRv4, Apache-2.0) backend for ``radintake.tools.ocr``. CPU only.

Selected with ``ocr.engine: paddle`` (or ``RADINTAKE_OCR_ENGINE=paddle``); needs the optional extra
``pip install -e .[paddle] -r requirements-paddle.lock``. Without it :func:`ocr` raises
``OcrUnavailable`` naming the extra - there is no silent fallback to Tesseract, so a run labelled
``paddleocr-ppocrv4`` was really read by PaddleOCR.

Contract (identical to the Tesseract backend): ``OcrWord`` with ``bbox`` in the pixel grid of the
image passed in, ``conf`` in 0..1 and ``block/par/line/word`` ids where ``line_id`` groups the words of
one text line, in reading order. PaddleOCR detects LINE boxes, so this module

1. takes each detected quad's axis-aligned box,
2. merges boxes whose vertical centres fall inside one another's span into one row (one ``line_id``,
   like Tesseract's ``--psm 6`` rows), ordered top to bottom and left to right,
3. splits each recognised line into words by whitespace and gives every word the slice of the line
   box proportional to its character offsets (uniform character advance), with the line confidence.

Models: PaddleOCR downloads the English PP-OCRv4 rec + PP-OCRv3 det weights to ``~/.paddleocr`` on first use. Under
``RADINTAKE_OFFLINE=1`` nothing is downloaded: the weights must already be there (prefetch with
``python -m radintake.tools.ocr_paddle --fetch``) or ``RADINTAKE_PADDLE_MODEL_DIR`` must point at a
folder holding ``det/`` and ``rec/`` inference models.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from radintake.config import env_flag
from radintake.core.types import BBox

ENGINE_ID = "paddleocr-ppocrv4"
EXTRA_HINT = "pip install -e .[paddle] -r requirements-paddle.lock"

_engine: Any = None
_version: str | None = None


def available() -> bool:
    """True when the ``paddle`` extra (paddleocr + paddlepaddle) is importable."""
    try:
        import paddle  # noqa: F401
        import paddleocr  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means the engine cannot run
        return False
    return True


def _unavailable(msg: str):
    from radintake.tools.ocr import OcrUnavailable

    return OcrUnavailable(msg)


def _model_dirs() -> dict[str, str]:
    """Explicit det/rec model folders, or {} to let PaddleOCR use its cache (~/.paddleocr)."""
    root = os.environ.get("RADINTAKE_PADDLE_MODEL_DIR")
    if root:
        d, r = Path(root) / "det", Path(root) / "rec"
        if not (d.is_dir() and r.is_dir()):
            raise _unavailable(f"RADINTAKE_PADDLE_MODEL_DIR={root} must contain det/ and rec/ model folders")
        return {"det_model_dir": str(d), "rec_model_dir": str(r)}
    if env_flag("RADINTAKE_OFFLINE"):
        cache = Path.home() / ".paddleocr" / "whl"
        if not (any(cache.glob("det/en/*/inference.pdiparams")) and any(cache.glob("rec/en/*/inference.pdiparams"))):
            raise _unavailable("PaddleOCR PP-OCRv4 weights are not cached and RADINTAKE_OFFLINE=1 forbids a "
                               "download: run `python -m radintake.tools.ocr_paddle --fetch` once online")
    return {}


def _get_engine():
    global _engine
    if _engine is None:
        if not available():
            raise _unavailable(f"ocr.engine=paddle but the PaddleOCR extra is not installed ({EXTRA_HINT})")
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        from paddleocr import PaddleOCR

        _engine = PaddleOCR(lang="en", ocr_version="PP-OCRv4", use_angle_cls=False, use_gpu=False,
                            enable_mkldnn=False, show_log=False, **_model_dirs())
    return _engine


def tool_version() -> str:
    """``paddleocr <version> PP-OCRv4 (paddle <version>)`` - recorded on every tool_call event."""
    global _version
    if _version is None:
        try:
            import paddle
            import paddleocr

            _version = f"paddleocr {getattr(paddleocr, '__version__', 'unknown')} PP-OCRv4 " \
                       f"(paddle {getattr(paddle, '__version__', 'unknown')})"
        except Exception:  # noqa: BLE001
            _version = "paddleocr unavailable"
    return _version


def _bgr(a: np.ndarray) -> np.ndarray:
    if a.ndim == 2:
        return np.ascontiguousarray(np.stack([a, a, a], axis=-1))
    return np.ascontiguousarray(a[:, :, ::-1])  # RGB (PIL/numpy convention here) -> BGR (paddle/cv2)


def _lines(raw: Any, w: int, h: int) -> list[tuple[BBox, str, float]]:
    """PaddleOCR result -> [(axis-aligned clamped box, text, conf)] for one image."""
    page = raw[0] if raw else None
    out: list[tuple[BBox, str, float]] = []
    for item in page or []:
        quad, (text, conf) = item[0], item[1]
        text = (text or "").strip()
        if not text:
            continue
        xs = [float(p[0]) for p in quad]
        ys = [float(p[1]) for p in quad]
        x0, y0 = max(0, int(np.floor(min(xs)))), max(0, int(np.floor(min(ys))))
        x1, y1 = min(w, int(np.ceil(max(xs)))), min(h, int(np.ceil(max(ys))))
        if x1 <= x0 or y1 <= y0:
            continue
        out.append((BBox(x0, y0, x1, y1), text, min(max(float(conf), 0.0), 1.0)))
    return out


def _rows(lines: list[tuple[BBox, str, float]]) -> list[list[tuple[BBox, str, float]]]:
    """Group line boxes into text rows (top to bottom), each row sorted left to right."""
    rows: list[list[tuple[BBox, str, float]]] = []
    spans: list[tuple[float, float]] = []
    for ln in sorted(lines, key=lambda t: ((t[0].y0 + t[0].y1) / 2.0, t[0].x0)):
        b = ln[0]
        cy = (b.y0 + b.y1) / 2.0
        for i, (top, bot) in enumerate(spans):
            if top <= cy <= bot:
                rows[i].append(ln)
                break
        else:
            rows.append([ln])
            spans.append((float(b.y0), float(b.y1)))
    for r in rows:
        r.sort(key=lambda t: t[0].x0)
    return rows


def split_words(box: BBox, text: str) -> list[tuple[str, BBox]]:
    """Split a recognised line into words, each with the proportional slice of the line box."""
    n = len(text)
    out: list[tuple[str, BBox]] = []
    i = 0
    width = box.x1 - box.x0
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace():
            j += 1
        x0 = box.x0 + round(width * i / n)
        x1 = max(x0 + 1, box.x0 + round(width * j / n))
        out.append((text[i:j], BBox(x0, box.y0, min(x1, box.x1), box.y1)))
        i = j
    return out


def ocr(page_image: Any, *, lang: str = "eng", trace: Any = None):
    """Same contract as ``radintake.tools.ocr.ocr`` (Tesseract); ``lang`` must be English."""
    from radintake.tools.ocr import OcrWord, _prep

    if lang not in ("eng", "en"):
        raise ValueError(f"paddle OCR backend is configured for English only (got lang={lang!r})")
    eng = _get_engine()
    t0 = time.perf_counter()
    arr = _prep(page_image)
    h, w = arr.shape[:2]
    raw = eng.ocr(_bgr(arr), det=True, rec=True, cls=False)
    words: list[OcrWord] = []
    for li, row in enumerate(_rows(_lines(raw, w, h)), start=1):
        wi = 0
        for box, text, conf in row:
            for t, wb in split_words(box, text):
                wi += 1
                words.append(OcrWord(text=t, bbox=wb, conf=conf, block=1, par=1, line=li, word=wi))
    if trace is not None:
        trace.tool_call("ocr", {"lang": lang, "w": int(w), "h": int(h)},
                        {"n_words": len(words),
                         "mean_conf": round(float(np.mean([x.conf for x in words])) if words else 0.0, 4)},
                        round((time.perf_counter() - t0) * 1000.0, 3), tool_version=tool_version())
    return words


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PaddleOCR backend utilities")
    ap.add_argument("--fetch", action="store_true", help="download/cache the PP-OCRv4 en det+rec weights")
    a = ap.parse_args(argv)
    if a.fetch:
        if env_flag("RADINTAKE_OFFLINE"):
            print("RADINTAKE_OFFLINE=1: refusing to download")
            return 1
        _get_engine()
        print(f"ready: {tool_version()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
