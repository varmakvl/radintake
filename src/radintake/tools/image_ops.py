"""Deterministic image metrics and geometry ops used by the preprocessing agent.

Every geometry op returns ``(new_image, Transform)`` where the transform maps coordinates of the
INPUT image to coordinates of the OUTPUT image (see ``radintake.core.types``). Ops never modify
their input array.
"""

from __future__ import annotations

import cv2
import numpy as np

from radintake.core.types import Transform


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# ---- metrics ---------------------------------------------------------------------------
def laplacian_variance(img: np.ndarray) -> float:
    """Blur metric: variance of the Laplacian (low = blurry)."""
    return float(cv2.Laplacian(to_gray(img), cv2.CV_64F).var())


def noise_sigma(img: np.ndarray) -> float:
    """Immerkaer fast noise-sigma estimate (grey levels)."""
    g = to_gray(img).astype(np.float64)
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(g, -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) * np.abs(conv).sum() / (6.0 * (w - 2) * (h - 2)))


def contrast(img: np.ndarray) -> float:
    """Robust contrast: 99th minus 1st percentile of grey levels (0..255); text pages are mostly paper."""
    g = to_gray(img)
    lo, hi = np.percentile(g, [1, 99])
    return float(hi - lo)


def _binary_ink(img: np.ndarray, max_side: int = 1000) -> np.ndarray:
    g = to_gray(img)
    h, w = g.shape
    s = max_side / max(h, w)
    if s < 1.0:
        g = cv2.resize(g, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b


def skew_angle(img: np.ndarray, max_angle: float = 15.0, step: float = 0.25) -> float:
    """Measured skew in degrees via projection profile; :func:`deskew` undoes it.

    Positive = text lines run counter-clockwise from horizontal (visually rising to the right).
    """
    b = _binary_ink(img)
    h, w = b.shape
    if b.sum() < 255 * 50:
        return 0.0
    c = (w / 2.0, h / 2.0)

    def score(a: float) -> float:
        m = cv2.getRotationMatrix2D(c, a, 1.0)
        r = cv2.warpAffine(b, m, (w, h), flags=cv2.INTER_NEAREST)
        prof = r.sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(prof) ** 2))

    angles = np.arange(-max_angle, max_angle + 1e-9, 1.0)
    best = max(angles, key=lambda a: (score(a), -abs(a)))
    fine = np.arange(best - 1.0, best + 1.0 + 1e-9, step)
    best = max(fine, key=lambda a: (score(a), -abs(a)))
    # cv2 positive angle = counter-clockwise rotation of the content that levels the lines
    return float(-best)


def find_page_quad(img: np.ndarray, *, min_area: float = 0.25, max_area: float = 0.97,
                   min_bg_contrast: float = 0.6) -> np.ndarray | None:
    """Corners (tl, tr, br, bl; 4x2 float) of a bright page on a darker background, else None.

    Returns None for scans where the page fills the frame, and when the bright region is not a
    convex quadrilateral. ``min_bg_contrast`` = max allowed (background mean / page mean).
    """
    g = to_gray(img)
    h, w = g.shape
    s = 800.0 / max(h, w)
    small = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else g
    sc = 1.0 / s if s < 1.0 else 1.0
    blur = cv2.GaussianBlur(small, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt) / float(small.shape[0] * small.shape[1])
    if not (min_area <= area <= max_area):
        return None
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float64)
            break
    if quad is None:
        return None
    inside = np.zeros_like(mask)
    cv2.fillConvexPoly(inside, quad.astype(np.int32), 255)
    page_px, bg_px = small[inside > 0], small[inside == 0]
    if bg_px.size == 0 or float(bg_px.mean()) > min_bg_contrast * max(float(page_px.mean()), 1.0):
        return None
    return order_quad(quad * sc)


def _page_edges(img: np.ndarray, max_side: float, close_px: int) -> tuple[np.ndarray, np.ndarray, float]:
    """(text-free smoothed grey, Canny edges, scale) at ``max_side`` resolution.

    A morphological closing wipes dark strokes narrower than ``close_px`` (text, table rules,
    handwriting), so only large-scale brightness steps - the paper edge - leave edges. A soft
    lighting gradient or shadow is smooth and leaves none.
    """
    g = to_gray(img)
    h, w = g.shape
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(g, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1.0 else g
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px))
    smooth = cv2.GaussianBlur(cv2.morphologyEx(small, cv2.MORPH_CLOSE, k), (5, 5), 0)
    edges = cv2.Canny(smooth, 8, 24, L2gradient=True)
    return smooth, edges, s


def _hough_lines(edges: np.ndarray, max_lines: int = 24) -> list[tuple[float, float]]:
    """Strongest distinct (rho, theta) lines, strongest first."""
    h, w = edges.shape
    raw = cv2.HoughLines(edges, 1, np.pi / 360, threshold=max(20, int(0.12 * min(h, w))))
    out: list[tuple[float, float]] = []
    if raw is None:
        return out
    for rho, theta in raw[:, 0, :]:
        dup = False
        for r2, t2 in out:
            dt = abs(theta - t2)
            if dt > np.pi / 2:  # theta wraps at pi with rho flipping sign
                dt, r2 = np.pi - dt, -r2
            if dt < np.deg2rad(6) and abs(rho - r2) < 12:
                dup = True
                break
        if not dup:
            out.append((float(rho), float(theta)))
            if len(out) >= max_lines:
                break
    return out


def _intersect(l1: tuple[float, float], l2: tuple[float, float]) -> np.ndarray | None:
    (r1, t1), (r2, t2) = l1, l2
    a = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    return np.linalg.solve(a, np.array([r1, r2]))


def _side_support(edges_d: np.ndarray, p0: np.ndarray, p1: np.ndarray, step: float = 3.0) -> float:
    """Fraction of points along segment p0-p1 that lie on a (dilated) edge pixel."""
    n = max(2, int(np.linalg.norm(p1 - p0) / step))
    t = np.linspace(0.04, 0.96, n)[:, None]  # skip the corners, where both sides meet
    pts = np.rint(p0 + (p1 - p0) * t).astype(int)
    h, w = edges_d.shape
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    if not ok.any():
        return 0.0
    hit = np.zeros(len(pts), bool)
    hit[ok] = edges_d[pts[ok, 1], pts[ok, 0]] > 0
    return float(hit.mean())


def _side_ratio(smooth: np.ndarray, p0: np.ndarray, p1: np.ndarray, centre: np.ndarray, off: float = 6.0) -> float:
    """Median grey just outside the side / median grey just inside it (local paper contrast)."""
    d = p1 - p0
    nrm = np.array([-d[1], d[0]]) / max(float(np.linalg.norm(d)), 1e-6)
    if np.dot(centre - (p0 + p1) / 2.0, nrm) < 0:
        nrm = -nrm  # point the normal into the page
    t = np.linspace(0.1, 0.9, 24)[:, None]
    base = p0 + d * t
    h, w = smooth.shape

    def sample(pts: np.ndarray) -> np.ndarray:
        p = np.rint(pts).astype(int)
        ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        return smooth[p[ok, 1], p[ok, 0]].astype(np.float64)

    inside, outside = sample(base + nrm * off), sample(base - nrm * off)
    if inside.size < 4 or outside.size < 4:
        return 1.0
    return float(np.median(outside) / max(float(np.median(inside)), 1.0))


def _refine_side(edges: np.ndarray, line: tuple[float, float], band: float = 4.0) -> tuple[float, float]:
    """Least-squares refit of a Hough line to the edge pixels within ``band`` px of it."""
    rho, theta = line
    ys, xs = np.nonzero(edges)
    if xs.size == 0:
        return line
    dist = np.abs(xs * np.cos(theta) + ys * np.sin(theta) - rho)
    sel = dist <= band
    if sel.sum() < 20:
        return line
    vx, vy, x0, y0 = cv2.fitLine(np.stack([xs[sel], ys[sel]], 1).astype(np.float32), cv2.DIST_HUBER,
                                 0, 0.01, 0.01).ravel()
    t = float(np.arctan2(vx, -vy)) % np.pi  # normal angle of the fitted direction
    r = float(x0 * np.cos(t) + y0 * np.sin(t))
    return r, t


def find_page_quad_lines(img: np.ndarray, *, min_area: float = 0.25, max_area: float = 0.97,
                         max_side_ratio: float = 0.88, min_support: float = 0.5,
                         min_side_support: float = 0.2, max_lines: int = 24,
                         close_px: int = 17) -> tuple[np.ndarray | None, float]:
    """Fallback page detector for low-contrast / shadowed phone photos: ``(quad or None, score)``.

    Works where :func:`find_page_quad` fails because no GLOBAL threshold separates paper from desk
    (a lighting gradient makes one side of the page darker than the desk on the other side). The
    paper edge is still a LOCAL brightness step, so: close away the text, take Canny edges, fit
    Hough lines, and score every quad made of two near-horizontal and two near-vertical lines by
    the fraction of its perimeter on an edge (``score``, 0..1). A quad is accepted only if it is
    convex, has near-rectangular corners, covers ``min_area..max_area`` of the frame and every
    side is a real paper edge: darker just outside than just inside
    (``outside / inside <= max_side_ratio``). A scan with no visible paper edge has no such quad.
    """
    smooth, edges, s = _page_edges(img, 800.0, close_px)
    h, w = edges.shape
    lines = _hough_lines(edges, max_lines)
    horiz = [ln for ln in lines if abs(np.sin(ln[1])) >= abs(np.cos(ln[1]))]
    vert = [ln for ln in lines if abs(np.sin(ln[1])) < abs(np.cos(ln[1]))]
    if len(horiz) < 2 or len(vert) < 2:
        return None, 0.0
    edges_d = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    frame = float(h * w)
    best, best_key = None, (-1.0, 0.0)
    pad_x, pad_y = 0.02 * w, 0.02 * h
    for i in range(len(horiz)):
        for j in range(i + 1, len(horiz)):
            for k in range(len(vert)):
                for m in range(k + 1, len(vert)):
                    pts = [_intersect(a, b) for a in (horiz[i], horiz[j]) for b in (vert[k], vert[m])]
                    if any(p is None for p in pts):
                        continue
                    q = order_quad(np.array(pts))
                    if (q[:, 0].min() < -pad_x or q[:, 0].max() > w + pad_x or q[:, 1].min() < -pad_y
                            or q[:, 1].max() > h + pad_y):
                        continue
                    if not cv2.isContourConvex(q.astype(np.float32).reshape(-1, 1, 2)):
                        continue
                    area = abs(cv2.contourArea(q.astype(np.float32))) / frame
                    if not (min_area <= area <= max_area):
                        continue
                    angles_ok = True
                    for c in range(4):
                        a, b = q[c - 1] - q[c], q[(c + 1) % 4] - q[c]
                        cosang = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6)
                        if abs(cosang) > 0.5:  # inner angle outside 60..120 degrees
                            angles_ok = False
                            break
                    if not angles_ok:
                        continue
                    sup = [_side_support(edges_d, q[c], q[(c + 1) % 4]) for c in range(4)]
                    score = float(np.mean(sup))
                    if score < min_support or min(sup) < min_side_support:
                        continue
                    key = (round(score, 2), area)  # best edge support, then the larger page
                    if key <= best_key:
                        continue
                    centre = q.mean(axis=0)
                    if max(_side_ratio(smooth, q[c], q[(c + 1) % 4], centre) for c in range(4)) > max_side_ratio:
                        continue
                    best, best_key = (horiz[i], horiz[j], vert[k], vert[m]), key
    if best is None:
        return None, 0.0
    ref = [_refine_side(edges, ln) for ln in best]
    pts = [_intersect(a, b) for a in ref[:2] for b in ref[2:]]
    if any(p is None for p in pts):
        pts = [_intersect(a, b) for a in best[:2] for b in best[2:]]
    q = order_quad(np.array(pts))
    return q / s, float(best_key[0])


def detect_page_quad(img: np.ndarray, p: dict) -> tuple[np.ndarray | None, str, float]:
    """Primary contour detector, then (if enabled) the line fallback: ``(quad, detector, score)``.

    ``detector`` is ``"contour"``, ``"lines"`` or ``"none"``; ``score`` is the edge support of
    the returned quad (0..1; for ``"contour"`` it is ``1 - background/page`` brightness).
    """
    quad = find_page_quad(img, min_area=p.get("quad_min_area", 0.25), max_area=p.get("quad_max_area", 0.97),
                          min_bg_contrast=p.get("quad_min_bg_contrast", 0.6))
    if quad is not None:
        return quad, "contour", _contour_score(img, quad)
    if p.get("quad_fallback", True):
        quad, score = find_page_quad_lines(img, min_area=p.get("quad_min_area", 0.25),
                                           max_area=p.get("quad_max_area", 0.97),
                                           max_side_ratio=p.get("quad_fallback_max_side_ratio", 0.88),
                                           min_support=p.get("quad_fallback_min_support", 0.5))
        if quad is not None:
            return quad, "lines", score
    return None, "none", 0.0


def _contour_score(img: np.ndarray, quad: np.ndarray) -> float:
    g = to_gray(img)
    inside = np.zeros(g.shape, np.uint8)
    cv2.fillConvexPoly(inside, np.rint(quad).astype(np.int32), 255)
    page_px, bg_px = g[inside > 0], g[inside == 0]
    if bg_px.size == 0 or page_px.size == 0:
        return 0.0
    return float(1.0 - float(bg_px.mean()) / max(float(page_px.mean()), 1.0))


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    s, d = pts.sum(axis=1), pts[:, 0] - pts[:, 1]
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]])


# ---- geometry ops ----------------------------------------------------------------------
def rotate_cw(img: np.ndarray, degrees: int) -> tuple[np.ndarray, Transform]:
    """Exact lossless clockwise rotation by a multiple of 90 degrees."""
    k = (-int(degrees) // 90) % 4
    h, w = img.shape[:2]
    return np.ascontiguousarray(np.rot90(img, k)), Transform.rot90(k, w, h)


def dewarp(img: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, Transform]:
    """4-point perspective crop of the page quad to an upright rectangle."""
    q = order_quad(quad).astype(np.float32)
    top, bot = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])
    left, right = np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])
    w, h = max(2, round(max(top, bot))), max(2, round(max(left, right)))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(q, dst).astype(np.float64)
    out = cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, Transform(m)


def deskew(img: np.ndarray, angle: float) -> tuple[np.ndarray, Transform]:
    """Rotate about the centre by ``-angle`` (undoing a measured skew); canvas size is kept."""
    h, w = img.shape[:2]
    t = Transform.rotation(-angle, w / 2.0, h / 2.0)
    out = cv2.warpPerspective(img, t.matrix, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return out, t


# ---- photometric ops (identity geometry) -------------------------------------------------
def denoise(img: np.ndarray, strength: float = 7.0) -> np.ndarray:
    g = to_gray(img)
    return cv2.fastNlMeansDenoising(g, None, h=float(strength), templateWindowSize=7, searchWindowSize=21)


def clahe(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    g = to_gray(img)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(g)


def _background(g: np.ndarray, k: int = 31, sigma: float = 15.0) -> np.ndarray:
    """Paper background estimate: closing removes dark text strokes, a wide blur smooths the rest."""
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.GaussianBlur(bg, (0, 0), sigma)


def illumination_unevenness(img: np.ndarray) -> float:
    """p95 - p5 of the background estimate (grey levels): ~1 for a flat scan, 30-70 for a lit photo."""
    bg = _background(to_gray(img))
    return float(np.percentile(bg, 95) - np.percentile(bg, 5))


def flatten_illumination(img: np.ndarray, white: float = 235.0) -> np.ndarray:
    """Flat-field correction: divide by the background so one global OCR threshold works again."""
    g = to_gray(img)
    out = np.clip(g.astype(np.float32) / np.maximum(_background(g), 1.0) * white, 0, 255).astype(np.uint8)
    return out
