"""Words from a PDF's own text layer, with boxes in the page's raster grid (no OCR).

A born-digital PDF already holds every character and its position, so OCR on its raster only adds
errors. ``text_layer_words`` reads the layer through pdfium and returns ``OcrWord``s (conf 1.0)
whose boxes are in ORIGINAL page coordinates: the page rasterised at ``dpi`` (PDF points x dpi/72,
y flipped), the same grid the loader produces - so the rest of the pipeline cannot tell the source.

Guards:

* **Hidden text.** Text a reader cannot see is a classic prompt-injection channel (white text,
  1-pt text, text placed off the page). A word is kept only if it lies on the page, its font is at
  least ``min_font_pt`` and the raster under its box carries ink. Dropped words are counted (never
  logged as text) and the page gets ``hidden_words`` > 0, which the caller turns into a review flag.
* **Mixed pages.** A digital header over a scanned body has a text layer that covers only part of
  the ink. ``uncovered_ink`` is the share of text-like ink on the raster outside every kept word box
  (ruled lines removed first); above ``mixed_ink_threshold`` the caller also OCRs the page and adds
  the OCR words that do not overlap a text-layer word.
* Rotated pages (``/Rotate`` != 0) and pages with fewer than ``min_chars`` visible characters fall
  back to OCR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from radintake.core.types import BBox
from radintake.tools.ocr import OcrWord

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "min_chars": 20,             # fewer visible characters on a page -> OCR the raster instead
    "min_font_pt": 4.0,          # smaller text is treated as hidden
    "ink_threshold": 160,        # gray level below which a pixel counts as ink
    "min_ink_fraction": 0.02,    # a visible word box has at least this share of ink pixels
    "mixed_ink_threshold": 0.2,  # uncovered text ink above this -> also OCR the page (mixed page)
    "word_gap_ratio": 0.15,      # horizontal gap > ratio x char height splits a word (loose boxes)
}


@dataclass
class TextLayer:
    words: list[OcrWord] = field(default_factory=list)  # visible words, original page coords
    n_chars: int = 0             # visible non-space characters
    hidden_words: int = 0        # dropped by the visibility guard
    uncovered_ink: float = 0.0   # share of text ink not covered by any kept word box
    usable: bool = False         # enough visible text to replace OCR
    mixed: bool = False          # usable, but the page also has text the layer does not cover
    reason: str = ""

    def stats(self) -> dict[str, Any]:
        """JSON-safe counts for page meta and traces (never text)."""
        return {"n_words": len(self.words), "n_chars": self.n_chars, "hidden_words": self.hidden_words,
                "uncovered_ink": round(self.uncovered_ink, 4), "usable": self.usable, "mixed": self.mixed,
                "reason": self.reason}


def _gray(raster: np.ndarray) -> np.ndarray:
    a = np.asarray(raster)
    if a.ndim == 3:
        a = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
    return a.astype(np.uint8) if a.dtype != np.uint8 else a


def _chars(textpage: Any, page_h_pt: float, scale: float) -> list[tuple[str, BBox, float, bool]]:
    """(char, raster box, font size pt, generated) per character in content order."""
    from pypdfium2 import raw

    out: list[tuple[str, BBox, float, bool]] = []
    for i in range(textpage.count_chars()):
        cp = raw.FPDFText_GetUnicode(textpage.raw, i)
        ch = chr(cp) if cp else ""
        generated = bool(raw.FPDFText_IsGenerated(textpage.raw, i) == 1)
        if not ch or ch.isspace() or generated:
            out.append((" ", BBox(0, 0, 0, 0), 0.0, True))
            continue
        # loose box = advance width x font ascent/descent: adjacent glyphs touch, so gaps mean spaces
        left, bottom, right, top = textpage.get_charbox(i, loose=True)
        x0, x1 = sorted((left * scale, right * scale))
        y0, y1 = sorted(((page_h_pt - top) * scale, (page_h_pt - bottom) * scale))
        out.append((ch, BBox(x0, y0, x1, y1), float(raw.FPDFText_GetFontSize(textpage.raw, i)), False))
    return out


def _words(chars: list[tuple[str, BBox, float, bool]], gap_ratio: float) -> list[tuple[str, BBox, float]]:
    words: list[tuple[str, BBox, float]] = []
    cur: list[tuple[str, BBox, float]] = []

    def flush() -> None:
        if cur:
            b = BBox(min(c[1].x0 for c in cur), min(c[1].y0 for c in cur),
                     max(c[1].x1 for c in cur), max(c[1].y1 for c in cur))
            words.append(("".join(c[0] for c in cur), b, min(c[2] for c in cur)))
            cur.clear()

    for ch, box, size, sep in chars:
        if sep:
            flush()
            continue
        if cur:
            prev = cur[-1][1]
            h = max(prev.height, box.height, 1.0)
            same_line = min(prev.y1, box.y1) - max(prev.y0, box.y0) > 0.3 * h
            if not same_line or box.x0 - prev.x1 > gap_ratio * h or box.x1 < prev.x0:
                flush()
        cur.append((ch, box, size))
    flush()
    return words


def assign_lines(words: list[tuple[str, BBox, float]]) -> list[OcrWord]:
    """Group words into lines by vertical overlap and give them Tesseract-style line ids."""
    lines: list[list[tuple[str, BBox, float]]] = []
    for w in sorted(words, key=lambda w: (w[1].y0, w[1].x0)):
        b = w[1]
        for ln in lines:
            r = ln[-1][1]
            if min(b.y1, r.y1) - max(b.y0, r.y0) > 0.5 * min(b.height, r.height):
                ln.append(w)
                break
        else:
            lines.append([w])
    lines.sort(key=lambda ln: min(w[1].y0 for w in ln))
    out: list[OcrWord] = []
    for li, ln in enumerate(lines, start=1):
        for wi, (text, box, conf) in enumerate(sorted(ln, key=lambda w: w[1].x0), start=1):
            out.append(OcrWord(text=text, bbox=box, conf=conf, block=1, par=1, line=li, word=wi))
    return out


def _text_ink(gray: np.ndarray, thr: int) -> np.ndarray:
    """Ink mask without long ruled lines (table borders are not text)."""
    import cv2

    ink = (gray < thr).astype(np.uint8)
    h, w = ink.shape
    hor = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 40), 1)))
    ver = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 40))))
    return ink & ~(hor | ver).astype(bool)


def text_layer_words(textpage: Any, page_size_pt: tuple[float, float], raster: np.ndarray, dpi: int,
                     params: dict[str, Any] | None = None) -> TextLayer:
    """Visible text-layer words of one page, boxes in the raster's pixel grid."""
    p = {**DEFAULTS, **(params or {})}
    scale = dpi / 72.0
    gray = _gray(raster)
    H, W = gray.shape[:2]
    tl = TextLayer()
    raw_words = _words(_chars(textpage, page_size_pt[1], scale), p["word_gap_ratio"])
    kept: list[tuple[str, BBox, float]] = []
    ink = gray < p["ink_threshold"]
    for text, box, size in raw_words:
        on_page = box.x1 > 0 and box.y1 > 0 and box.x0 < W and box.y0 < H and box.width > 0 and box.height > 0
        visible = on_page and size >= p["min_font_pt"]
        if visible:
            b = box.clip(W, H)
            x0, y0, x1, y1 = int(b.x0), int(b.y0), int(np.ceil(b.x1)), int(np.ceil(b.y1))
            region = ink[y0:y1, x0:x1]
            visible = region.size > 0 and float(region.mean()) >= p["min_ink_fraction"]
        if visible:
            kept.append((text, box.clip(W, H), 1.0))
        else:
            tl.hidden_words += 1
    tl.words = assign_lines(kept)
    tl.n_chars = sum(len(w.text) for w in tl.words)
    if tl.n_chars < p["min_chars"]:
        tl.reason = "too_few_chars"
        return tl
    text_ink = _text_ink(gray, p["ink_threshold"])
    total = int(text_ink.sum())
    if total:
        covered = np.zeros_like(text_ink)
        for w in tl.words:
            b = w.bbox
            covered[max(0, int(b.y0) - 3):int(b.y1) + 4, max(0, int(b.x0) - 3):int(b.x1) + 4] = True
        tl.uncovered_ink = float((text_ink & ~covered).sum()) / total
    tl.usable = True
    tl.mixed = tl.uncovered_ink > p["mixed_ink_threshold"]
    tl.reason = "mixed_page" if tl.mixed else "digital"
    return tl


def merge_ocr(text_words: list[OcrWord], ocr_words: list[OcrWord]) -> list[OcrWord]:
    """Text-layer words plus the OCR words that overlap none of them (mixed pages), re-lined."""
    def overlaps(a: BBox) -> bool:
        return any(min(a.x1, b.bbox.x1) - max(a.x0, b.bbox.x0) > 0 and
                   min(a.y1, b.bbox.y1) - max(a.y0, b.bbox.y0) > 0 for b in text_words)

    extra = [(w.text, w.bbox, w.conf) for w in ocr_words if not overlaps(w.bbox)]
    return assign_lines([(w.text, w.bbox, w.conf) for w in text_words] + extra)


def page_words(page: Any, *, trace: Any = None) -> list[OcrWord]:
    """Words for a (possibly processed) page: text layer when usable, else / plus OCR.

    Text-layer words are stored in original coordinates and mapped to the page's current image.
    """
    from radintake.tools.ocr import ocr

    tl: TextLayer | None = getattr(page, "text_layer", None)
    if tl is None or not tl.usable:
        return ocr(page.image, trace=trace)
    mapped = [OcrWord(text=w.text, bbox=page.to_current(w.bbox), conf=w.conf, block=w.block, par=w.par,
                      line=w.line, word=w.word) for w in tl.words]
    if trace is not None:
        trace.tool_call("pdf_text_layer", {"page": page.index}, tl.stats(), 0.0, tool_version="pdfium")
    if tl.mixed:
        return merge_ocr(mapped, ocr(page.image, trace=trace))
    return mapped
