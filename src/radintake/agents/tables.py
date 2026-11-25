"""Table stage: page words (PDF text layer or OCR) + table detection, before PHI and extraction.

Stage ``tables`` (order 25, after preprocessing). For each page it

1. gets the words once (``tools.pdf_text.page_words``: text layer when the PDF has one, else
   Tesseract) and caches them as ``ocr_words`` so the PHI, extraction and verify stages reuse them;
2. detects tables (``tools.tables``), re-reading each ruled cell on OCR pages;
3. writes re-read cell text back into the cached words (the cell's old words are replaced), so PHI
   masking, field extraction and the verifier's grounding check all see the better reading;
4. returns ``tables`` for the output (cell boxes in ORIGINAL page coordinates) and ``table_objs``
   (processed coordinates) for the extraction stage, which keeps table text out of section prose
   and records the section each table sits in.

Traces get counts only, never cell text (cells can hold PHI).
"""

from __future__ import annotations

import time
from typing import Any

from radintake.core.types import BBox, Page
from radintake.orchestrator.stages import DOCUMENT_TYPES, StageContext, register_stage
from radintake.tools import tables as T
from radintake.tools.ocr import OcrWord


def load_params(cfg: Any) -> dict[str, Any]:
    raw = getattr(cfg, "raw", None) or {}
    user = raw.get("tables") if isinstance(raw.get("tables"), dict) else {}
    return {**T.DEFAULTS, **user}


def _cell_words(text: str, box: BBox, conf: float, line_id: tuple[int, int, int]) -> list[OcrWord]:
    """Split re-read cell text into words with boxes spread over the cell by character count."""
    toks = text.split()
    total = sum(len(t) for t in toks) + max(0, len(toks) - 1)
    out, x = [], box.x0
    unit = box.width / max(1, total)
    for i, t in enumerate(toks):
        x1 = x + unit * len(t)
        out.append(OcrWord(text=t, bbox=BBox(x, box.y0, max(x1, x + 1), box.y1), conf=conf,
                           block=line_id[0], par=line_id[1], line=line_id[2], word=i + 1))
        x = x1 + unit
    return out


def apply_cells(words: list[Any], tables: list[T.Table], changed: set[tuple[int, int, int]]) -> list[Any]:
    """Replace the words inside re-read cells with the cell's text (``changed`` = (table, row, col))."""
    if not changed:
        return words
    out = list(words)
    for ti, t in enumerate(tables):
        for c in t.cells:
            if (ti, c.row, c.col) not in changed:
                continue
            inside = [w for w in out if T._center_in(w.bbox, c.bbox)]
            # own line id per table row: OCR line ids around a table often run into neighbouring text
            line_id = (900 + ti, c.row + 1, 1)
            inner = BBox(c.bbox.x0 + 4, c.bbox.y0 + 4, max(c.bbox.x0 + 5, c.bbox.x1 - 4),
                         max(c.bbox.y0 + 5, c.bbox.y1 - 4))
            ids = {id(w) for w in inside}
            if c.reread_words:  # Tesseract's own word boxes from the cell crop (tight, like page OCR)
                new = [OcrWord(text=t, bbox=b, conf=cf, block=line_id[0], par=line_id[1], line=line_id[2],
                               word=i + 1) for i, (t, b, cf) in enumerate(c.reread_words)]
            else:
                new = _cell_words(c.text, inner, c.confidence, line_id)
            out = [w for w in out if id(w) not in ids] + new
    return out


def table_output(t: T.Table, page: Page) -> dict[str, Any]:
    """Output form of one table: boxes mapped to ORIGINAL page coordinates."""
    def orig(b: BBox) -> list[float]:
        return [round(v, 2) for v in page.to_original(b).as_list()]

    return {"page": page.index, "bbox": orig(t.bbox), "n_rows": t.n_rows, "n_cols": t.n_cols,
            "method": t.method, "text_source": "pdf_text" if page.text_layer is not None else "ocr",
            "confidence": t.confidence, "section": None,
            "cells": [{"row": c.row, "col": c.col, "text": c.text, "bbox": orig(c.bbox),
                       "confidence": c.confidence} for c in t.cells]}


@register_stage("tables", applies_to=DOCUMENT_TYPES, order=25)  # a failure -> review reason "tables: ..."
def tables_stage(ctx: StageContext) -> dict[str, Any]:
    from radintake.tools.ocr import ocr
    from radintake.tools.pdf_text import page_words

    p = load_params(ctx.config)
    pages: list[Page] = list(ctx.data.get("pages") or [])
    t0 = time.perf_counter()
    words_by_page: dict[int, list[Any]] = {}
    out_tables: list[dict[str, Any]] = []
    objs: dict[int, list[T.Table]] = {}
    for pg in pages:
        words = page_words(pg, trace=ctx.trace)
        ocr_fn = None if (pg.text_layer is not None and not pg.text_layer.mixed) else (
            lambda img, psm=7: ocr(img, psm=psm))
        before: dict[tuple[int, int, int], str] = {}
        found = T.detect_tables(pg.image, words, p, ocr_fn=None)
        for ti, t in enumerate(found):
            for c in t.cells:
                before[(ti, c.row, c.col)] = c.text
        if ocr_fn is not None and p.get("cell_reread", True):
            for t in found:
                if t.method == "ruled":
                    T._reread(T._gray(pg.image), t, p, ocr_fn)
        changed = {k for k in before
                   if found[k[0]].cells[k[1] * found[k[0]].n_cols + k[2]].text != before[k]}
        words = apply_cells(words, found, changed)
        found = [t for t in found if t.filled_fraction > 0]  # a ruled grid nothing could be read from
        words_by_page[pg.index] = words
        objs[pg.index] = found
        out_tables += [table_output(t, pg) for t in found]
        ctx.trace.event("tables_page", page=pg.index, n_words=len(words), n_tables=len(found),
                        shapes=[[t.n_rows, t.n_cols] for t in found], methods=[t.method for t in found],
                        cells_reread=len(changed), confidences=[t.confidence for t in found])
    ctx.trace.event("tables_done", n_pages=len(pages), n_tables=len(out_tables), version=T.TABLES_VERSION,
                    duration_ms=round((time.perf_counter() - t0) * 1000.0, 3))
    return {"ocr_words": words_by_page, "tables": out_tables, "table_objs": objs}
