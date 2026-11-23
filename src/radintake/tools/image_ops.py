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
