"""Table detection and cell reading from a page image plus its words. CPU only, no model.

Two detectors, both returning cells with boxes in the coordinates of the image passed in:

* **ruled** - long horizontal rules found by morphology. Rules that stack (overlapping x extent,
  small vertical gaps) form one table. Column borders are vertical rules inside the table that are
  longer than a text line. Without vertical rules (rules-only / "booktabs" tables), columns come
  from the x-projection of the words: an empty vertical strip wider than ``col_gap_ratio`` x the
  median word height separates two columns. Fully ruled tables keep one row per band (a wrapped
  cell stays one cell); other tables split bands into one row per text line.
* **aligned** - borderless tables: runs of >= ``min_rows`` consecutive text lines that each break
  into >= ``min_cols`` gap-separated segments, with columns again from the x-projection.

Words come from the PDF text layer or OCR (``tools.pdf_text.page_words``); a cell's confidence is
the mean confidence of its words. Limits (v1): no merged / spanning cells, no nested tables,
no rotated tables.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from radintake.core.types import BBox

TABLES_VERSION = "tables.v1"

DEFAULTS: dict[str, Any] = {
    "ink_threshold": 160,     # gray below this = ink (clean / digital pages)
    "min_rule_frac": 0.12,    # a horizontal rule spans at least this share of the page width
    "max_band_ratio": 16.0,   # rules further apart than this x word height do not stack (a rules-only
                              # table's body band holds all its rows; prose between page rules is
                              # rejected later because it has no column gaps)
    "edge_tol_px": 12,        # stacked rules' left / right ends agree within this, 0.6 x word height
    "edge_tol_frac": 0.02,    # or this share of the page width, whichever is largest
    "col_gap_ratio": 1.2,     # empty strip wider than this x word height splits columns
    "seg_gap_ratio": 1.2,     # aligned detector: gap > this x line height splits a line into cells
    "min_cols": 2,            # ruled tables with vertical rules
    "min_cols_unruled": 3,    # rules-only and aligned tables (fewer columns = key/value text, not a table)
    "min_rows": 2,
    "min_filled": 0.5,        # aligned: share of cells that must hold text
    "min_aligned_conf": 0.6,  # aligned tables read with lower mean confidence are noise, not tables
    "cell_reread": True,      # OCR pages: re-read each ruled cell on its own (borders removed, psm 7)
    "cell_inset_px": 4,       # crop inside the cell borders
    "cell_min_height_px": 40, # smaller crops are upscaled to this height before OCR
}


@dataclass
class Cell:
    row: int
    col: int
    text: str
    bbox: BBox
    confidence: float
    # (text, box in page coords, conf) of a per-cell re-read, so callers can keep tight word boxes
    reread_words: list[tuple[str, BBox, float]] | None = None


@dataclass
class Table:
    bbox: BBox
    n_rows: int
    n_cols: int
    method: str                       # ruled | rules_only | aligned
    cells: list[Cell] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        filled = [c.confidence for c in self.cells if c.text]
        return round(statistics.fmean(filled), 4) if filled else 0.0

    @property
    def filled_fraction(self) -> float:
        return sum(1 for c in self.cells if c.text) / max(1, self.n_rows * self.n_cols)

    def rows(self) -> list[list[str]]:
        grid = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for c in self.cells:
            grid[c.row][c.col] = c.text
        return grid


# --------------------------------------------------------------------------- helpers
def _gray(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 3:
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return np.clip(a, 0, 255).astype(np.uint8)


def _ink(gray: np.ndarray, thr: int) -> np.ndarray:
    """Ink mask: global threshold on clean pages, adaptive on uneven (scanned / photographed) ones."""
    import cv2

    if float(np.percentile(gray, 95)) - float(np.percentile(gray, 50)) < 8 and float(np.median(gray)) > 200:
        return (gray < thr).astype(np.uint8)
    return (cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)).astype(np.uint8)


def _segments(mask: np.ndarray, horizontal: bool) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (x0, y0, x1, y1) of connected line components."""
    import cv2

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i, :4])
        if (horizontal and w > 4 * h) or (not horizontal and h > 4 * w):
            out.append((x, y, x + w, y + h))
    return out


def _center_in(b: BBox, r: BBox) -> bool:
    cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
    return r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1


def _lines_of(words: list[Any]) -> list[list[Any]]:
    """Words grouped into text lines by vertical overlap, sorted top-down, left-right."""
    lines: list[list[Any]] = []
    for w in sorted(words, key=lambda w: (w.bbox.y0, w.bbox.x0)):
        for ln in lines:
            r = ln[-1].bbox
            if min(w.bbox.y1, r.y1) - max(w.bbox.y0, r.y0) > 0.5 * min(w.bbox.height, r.height):
                ln.append(w)
                break
        else:
            lines.append([w])
    for ln in lines:
        ln.sort(key=lambda w: w.bbox.x0)
    lines.sort(key=lambda ln: min(w.bbox.y0 for w in ln))
    return lines


def _projection_cols(words: list[Any], x0: float, x1: float, min_gap: float) -> list[float]:
    """Column borders (including x0 and x1) from empty vertical strips between words."""
    spans = sorted((w.bbox.x0, w.bbox.x1) for w in words)
    if not spans:
        return [x0, x1]
    borders = [x0]
    cur_end = spans[0][1]
    for a, b in spans[1:]:
        if a - cur_end > min_gap:
            borders.append((a + cur_end) / 2)
        cur_end = max(cur_end, b)
    borders.append(x1)
    return borders


def _clean(text: str) -> str:
    """Drop border glyphs OCR reads off cell rules ("|", "[", "]" at the ends)."""
    return text.strip().strip("|[]{}_").strip()


def _join(words: list[Any]) -> str:
    return _clean(" ".join(" ".join(w.text for w in ln) for ln in _lines_of(words)))


def _fill(table_box: BBox, row_bands: list[tuple[float, float]], col_borders: list[float],
          words: list[Any], method: str) -> Table:
    cells: list[Cell] = []
    for r, (y0, y1) in enumerate(row_bands):
        for c in range(len(col_borders) - 1):
            box = BBox(col_borders[c], y0, col_borders[c + 1], y1)
            inside = [w for w in words if _center_in(w.bbox, box)]
            conf = statistics.fmean(float(getattr(w, "conf", 1.0)) for w in inside) if inside else 0.0
            cells.append(Cell(r, c, _join(inside), box, round(conf, 4)))
    return Table(table_box, len(row_bands), len(col_borders) - 1, method, cells)


# --------------------------------------------------------------------------- ruled
def _ruled(gray: np.ndarray, words: list[Any], p: dict[str, Any], wh: float) -> list[Table]:
    import cv2

    W = gray.shape[1]
    ink = _ink(gray, p["ink_threshold"])
    kw = max(30, int(p["min_rule_frac"] * W))
    hor = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1)))
    hsegs = sorted(_segments(hor, True), key=lambda s: s[1])
    # stack rules into table candidates
    groups: list[list[tuple[int, int, int, int]]] = []
    for s in hsegs:
        for g in groups:
            last = g[-1]
            # rules of one table share their left and right ends (a slightly slanted scan loses a little)
            tol = max(p["edge_tol_px"], 0.6 * wh, p["edge_tol_frac"] * W)
            same_edges = abs(s[0] - last[0]) <= tol and abs(s[2] - last[2]) <= tol
            if same_edges and s[1] - last[3] <= p["max_band_ratio"] * wh:
                g.append(s)
                break
        else:
            groups.append([s])
    kh = max(10, int(1.3 * wh))
    ver = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh)))
    vsegs = _segments(ver, False)
    tables: list[Table] = []
    for g in groups:
        if len(g) < 2:
            continue
        # merge rules that are really one thick line (a few px apart)
        ys: list[tuple[float, float]] = []
        for s in g:
            if ys and s[1] - ys[-1][1] <= 4:
                ys[-1] = (ys[-1][0], s[3])
            else:
                ys.append((s[1], s[3]))
        if len(ys) < 2:
            continue
        x0, x1 = min(s[0] for s in g), max(s[2] for s in g)
        y0, y1 = ys[0][0], ys[-1][1]
        # no "words inside" check here: Tesseract often returns nothing for a dense ruled grid, which
        # the per-cell re-read then reads; a grid that stays empty is dropped after the re-read
        inside = [w for w in words if _center_in(w.bbox, BBox(x0, y0, x1, y1))]
        # table rules end at the table's outer rules; scratches / page lines running past them do not count
        vtab = [s for s in vsegs if s[0] >= x0 - 6 and s[2] <= x1 + 6 and s[1] >= y0 - 6 and s[3] <= y1 + 6]
        vin = [sv for sv in vtab if x0 + 8 < (sv[0] + sv[2]) / 2 < x1 - 8]
        bands = [(ys[k][1], ys[k + 1][0]) for k in range(len(ys) - 1)]

        def words_in(b: tuple[float, float], inside: list = inside) -> list[Any]:
            return [w for w in inside if b[0] <= (w.bbox.y0 + w.bbox.y1) / 2 <= b[1]]

        def crossed(b: tuple[float, float], vin: list = vin) -> bool:
            return any(min(sv[3], b[1]) - max(sv[1], b[0]) >= 0.5 * (b[1] - b[0]) for sv in vin)

        def tabular(b: tuple[float, float], x0: float = x0, x1: float = x1, vin: list = vin) -> bool:
            """Table-like band: crossed by a vertical rule when the stack has any (a key/value block
            between two ruled tables has columns too, but no rules); else its words form columns."""
            if vin:  # a crossed band counts even when OCR read nothing in it (shaded header rows)
                return crossed(b)
            bw = words_in(b)
            return bool(bw) and len(_projection_cols(bw, x0, x1, p["col_gap_ratio"] * wh)) - 1 >= p["min_cols_unruled"]

        # split the stack at bands that are not table-like (prose between two same-width tables,
        # a separator between double rules); each run of table-like bands is one table
        runs: list[list[tuple[float, float]]] = [[]]
        for b in bands:
            tall_enough = b[1] - b[0] > 0.6 * wh or (b[1] - b[0] > 8 and (words_in(b) or crossed(b)))
            if tall_enough and tabular(b):
                runs[-1].append(b)
            elif runs[-1]:
                runs.append([])
        for run in runs:
            t = _build_ruled(run, x0, x1, inside, vin, p, wh)
            if t is not None:
                tables.append(t)
    return tables


def _build_ruled(run: list[tuple[float, float]], x0: float, x1: float, words: list[Any],
                 vin: list[tuple[int, int, int, int]], p: dict[str, Any], wh: float) -> Table | None:
    if not run:
        return None
    y0, y1 = run[0][0], run[-1][1]
    tbox = BBox(x0, y0, x1, y1)
    inside = [w for w in words if _center_in(w.bbox, tbox)]
    inner: list[float] = []
    for x in sorted((sv[0] + sv[2]) / 2 for sv in vin if sv[3] > y0 + 4 and sv[1] < y1 - 4):
        if not inner or x - inner[-1] > 8:
            inner.append(x)
    bands = list(run)
    if inner:
        borders, method, min_cols = [x0, *inner, x1], "ruled", p["min_cols"]
    else:
        if not inside:
            return None
        borders = _projection_cols(inside, x0, x1, p["col_gap_ratio"] * wh)
        method, min_cols = "rules_only", p["min_cols_unruled"]
    # one row per text line when the rules do not separate rows: rules-only tables, and ruled tables
    # with a single band (each column boxed, label and value stacked inside the box)
    if method == "rules_only" or len(bands) == 1:
        rows: list[tuple[float, float]] = []
        for b0, b1 in bands:
            lines = _lines_of([w for w in inside if b0 <= (w.bbox.y0 + w.bbox.y1) / 2 <= b1])
            if len(lines) <= 1:
                rows.append((b0, b1))
                continue
            cuts = [b0] + [(max(w.bbox.y1 for w in lines[k]) + min(w.bbox.y0 for w in lines[k + 1])) / 2
                           for k in range(len(lines) - 1)] + [b1]
            rows += [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]
        bands = rows
    if inside:  # drop empty sliver columns at the edges (rules running a little past the outer border)
        def empty(a: float, b: float) -> bool:
            return not any(a <= (w.bbox.x0 + w.bbox.x1) / 2 <= b for w in inside)

        while len(borders) > 2 and borders[-1] - borders[-2] < 1.5 * wh and empty(borders[-2], borders[-1]):
            borders.pop()
        while len(borders) > 2 and borders[1] - borders[0] < 1.5 * wh and empty(borders[0], borders[1]):
            borders.pop(0)
        tbox = BBox(borders[0], tbox.y0, borders[-1], tbox.y1)
    if len(borders) - 1 < min_cols or len(bands) < p["min_rows"]:
        return None
    return _fill(tbox, bands, borders, inside, method)


# --------------------------------------------------------------------------- aligned
def _aligned(words: list[Any], taken: list[BBox], p: dict[str, Any]) -> list[Table]:
    free = [w for w in words if not any(_center_in(w.bbox, t) for t in taken)]
    lines = _lines_of(free)

    def segments(ln: list[Any]) -> list[list[Any]]:
        h = statistics.median(w.bbox.height for w in ln)
        segs: list[list[Any]] = [[ln[0]]]
        for w in ln[1:]:
            if w.bbox.x0 - segs[-1][-1].bbox.x1 > p["seg_gap_ratio"] * h:
                segs.append([w])
            else:
                segs[-1].append(w)
        return segs

    runs: list[list[list[Any]]] = []
    cur: list[list[Any]] = []
    for ln in lines:
        if len(segments(ln)) < p["min_cols_unruled"]:
            if cur:
                runs.append(cur)
            cur = []
            continue
        h = statistics.median(w.bbox.height for w in ln)
        if cur and min(w.bbox.y0 for w in ln) - max(w.bbox.y1 for w in cur[-1]) > 2.5 * h:
            runs.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        runs.append(cur)
    tables: list[Table] = []
    for run in runs:
        if len(run) < p["min_rows"]:
            continue
        ws = [w for ln in run for w in ln]
        wh = statistics.median(w.bbox.height for w in ws)
        x0, x1 = min(w.bbox.x0 for w in ws) - 2, max(w.bbox.x1 for w in ws) + 2
        borders = _projection_cols(ws, x0, x1, p["col_gap_ratio"] * wh)
        if len(borders) - 1 < p["min_cols_unruled"]:
            continue
        tops = [min(w.bbox.y0 for w in ln) for ln in run]
        bots = [max(w.bbox.y1 for w in ln) for ln in run]
        bands = [((bots[k - 1] + tops[k]) / 2 if k else tops[k] - 2,
                  (bots[k] + tops[k + 1]) / 2 if k + 1 < len(run) else bots[k] + 2) for k in range(len(run))]
        t = _fill(BBox(x0, bands[0][0], x1, bands[-1][1]), bands, borders, ws, "aligned")
        if t.filled_fraction >= p["min_filled"]:
            tables.append(t)
    return tables


# --------------------------------------------------------------------------- cell re-read (OCR pages)
def _reread_cell(gray: np.ndarray, box: BBox, p: dict[str, Any],
                 ocr_fn: Any) -> tuple[str, float, list[tuple[str, BBox, float]]]:
    """OCR one cell crop as a single text line: inset past the borders, Otsu, upscale if small.

    Returns text, mean confidence and the words with boxes mapped back to page coordinates."""
    import cv2

    k = p["cell_inset_px"]
    x0, y0, x1, y1 = int(box.x0) + k, int(box.y0) + k, int(box.x1) - k, int(box.y1) - k
    if x1 - x0 < 6 or y1 - y0 < 6:
        return "", 0.0, []
    crop = gray[y0:y1, x0:x1]
    _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw < 128).mean() > 0.5:  # light text on a dark (shaded) cell -> make the text dark
        bw = 255 - bw
    fx = fy = 1.0
    if bw.shape[0] < p["cell_min_height_px"]:
        fy = p["cell_min_height_px"] / bw.shape[0]
        new_w = int(bw.shape[1] * fy)
        fx = new_w / bw.shape[1]
        bw = cv2.resize(bw, (new_w, p["cell_min_height_px"]), interpolation=cv2.INTER_CUBIC)
    pad = 10
    bw = cv2.copyMakeBorder(bw, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    ws = [w for w in ocr_fn(bw, psm=7) if w.text.strip()]
    if not ws:
        return "", 0.0, []

    def back(b: BBox) -> BBox:  # crop (padded, scaled) -> page coordinates, clipped to the cell
        return BBox(min(max(x0 + (b.x0 - pad) / fx, x0), x1 - 1), min(max(y0 + (b.y0 - pad) / fy, y0), y1 - 1),
                    min(max(x0 + (b.x1 - pad) / fx, x0 + 1), x1), min(max(y0 + (b.y1 - pad) / fy, y0 + 1), y1))

    mapped = [(w.text, back(w.bbox), float(w.conf)) for w in ws]
    return " ".join(w.text for w in ws), statistics.fmean(w.conf for w in ws), mapped


def _reread(gray: np.ndarray, t: Table, p: dict[str, Any], ocr_fn: Any) -> None:
    for c in t.cells:
        text, conf, mapped = _reread_cell(gray, c.bbox, p, ocr_fn)
        text = _clean(text)
        if text and conf > c.confidence:
            c.text, c.confidence = text, round(conf, 4)
            c.reread_words = [(_clean(t), b, cf) for t, b, cf in mapped if _clean(t)]


def detect_tables(image: np.ndarray, words: list[Any], params: dict[str, Any] | None = None, *,
                  ocr_fn: Any = None) -> list[Table]:
    """Tables on one page; boxes in the coordinates of ``image`` (and of ``words``).

    ``ocr_fn(img, psm=...)`` (Tesseract) enables the per-cell re-read; pass None for text-layer
    pages, whose words are exact already.
    """
    p = {**DEFAULTS, **(params or {})}
    words = [w for w in words if (w.text or "").strip()]
    if not words:
        return []
    wh = float(statistics.median(w.bbox.height for w in words))
    gray = _gray(image)
    tables = _ruled(gray, words, p, wh)
    if ocr_fn is not None and p["cell_reread"]:
        for t in tables:
            if t.method == "ruled":
                _reread(gray, t, p, ocr_fn)
    tables = [t for t in tables if t.filled_fraction > 0] if ocr_fn is not None else tables
    aligned = [t for t in _aligned(words, [t.bbox for t in tables], p) if t.confidence >= p["min_aligned_conf"]]
    tables = sorted(tables + aligned, key=lambda t: (t.bbox.y0, t.bbox.x0))
    return tables
