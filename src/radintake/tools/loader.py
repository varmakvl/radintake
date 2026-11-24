"""Page loader: turns a routed input file into ``Page`` objects.

Coordinate conventions (see ``radintake.core.types``):

* PNG / JPEG: the original page is the pixel grid as stored in the file. EXIF orientation is
  NOT applied here (it is recorded in ``page.meta["exif_orientation"]``); fixing orientation is
  the preprocessing agent's job and is tracked as a ``Transform`` so boxes map back.
* TIFF: every frame becomes one page, pixel grid as stored.
* PDF: each page is rasterised at ``dpi`` (default 200, ``loader.pdf_dpi``). Page coordinates
  in every output are the RASTERISED PIXEL coordinates at that DPI (PDF point coords x dpi/72),
  recorded in ``page.meta`` (``dpi``, ``pdf_size_pt``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

from radintake.core.types import Page

LOADER_VERSION = "loader.v1"
_EXIF_ORIENTATION = 0x0112


def _to_array(img: Image.Image) -> np.ndarray:
    if img.mode not in ("L", "RGB"):
        img = img.convert("L") if img.mode in ("1", "I;16", "I", "F", "LA") else img.convert("RGB")
    return np.asarray(img).copy()


def _page(index: int, arr: np.ndarray, source: str, **meta) -> Page:
    h, w = arr.shape[:2]
    return Page(index=index, image=arr, original_size=(int(w), int(h)), source=source, meta=meta)


def load_image(path: str | Path) -> list[Page]:
    with Image.open(path) as img:
        exif = img.getexif() if hasattr(img, "getexif") else {}
        orientation = int(exif.get(_EXIF_ORIENTATION, 1)) if exif else 1
        fmt = (img.format or "image").lower()
        arr = _to_array(img)
    return [_page(1, arr, fmt, exif_orientation=orientation)]


def load_tiff(path: str | Path) -> list[Page]:
    pages: list[Page] = []
    with Image.open(path) as img:
        for i, frame in enumerate(ImageSequence.Iterator(img), start=1):
            pages.append(_page(i, _to_array(frame), "tiff-frame"))
    if not pages:
        raise ValueError("TIFF has no frames")
    return pages


def load_pdf(path: str | Path, dpi: int = 200) -> list[Page]:
    import pypdfium2 as pdfium

    pages: list[Page] = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            try:
                w_pt, h_pt = page.get_size()
                pil = page.render(scale=dpi / 72.0).to_pil()
            finally:
                page.close()
            pages.append(_page(i + 1, _to_array(pil), f"pdf@{dpi}dpi", dpi=dpi,
                               pdf_size_pt=[float(w_pt), float(h_pt)]))
    finally:
        pdf.close()
    if not pages:
        raise ValueError("PDF has no pages")
    return pages


def load_pages(path: str | Path, input_type: str, *, dpi: int = 200) -> list[Page]:
    """Load every page of a routed document input (scan, photo, pdf, tiff)."""
    if input_type in ("scan", "photo"):
        return load_image(path)
    if input_type == "tiff":
        return load_tiff(path)
    if input_type == "pdf":
        return load_pdf(path, dpi=dpi)
    raise ValueError(f"load_pages does not handle input_type={input_type!r}")
