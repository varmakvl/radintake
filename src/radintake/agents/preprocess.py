"""Preprocessing agent: quality metrics drive orientation, dewarp, deskew, denoise and contrast.

Why a rule-based policy and not an LLM: the decisions are small, numeric, cheap and must be
reproducible and explainable per page (``decision <- metric > threshold``). An LLM would add
latency, nondeterminism and a PHI exposure surface for no accuracy gain.

Policy (in order, each step only if its metric demands it)::

    page quad detected      -> 4-point perspective dewarp (photos)
    background unevenness   -> flat-field illumination correction (lighting gradient, shadow)
      > illumination_uneven_max
    orientation (OCR tool)  -> rotate by 90/180/270 (lossless), judged on the flat, even page
    |skew| > deskew_min_deg -> rotate by the measured skew
    noise > noise_sigma_max -> non-local-means denoise
    contrast < contrast_min -> CLAHE

Work happens on copies: the source file is never opened for writing and the loaded page array is
never mutated. The composed ``Page.transform`` (T_page, original -> processed) is kept, so any
box found on the processed page maps back with :func:`map_to_original`.
Decisions and the metric values that triggered them are recorded in the trace (numbers only).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from radintake.core.types import BBox, Page, Transform
from radintake.orchestrator.stages import DOCUMENT_TYPES, StageContext, register_stage
from radintake.tools import image_ops as ops

DEFAULTS: dict[str, Any] = {
    "deskew_min_deg": 0.3,       # ignore skew below this
    "deskew_max_deg": 15.0,      # search range
    "noise_sigma_max": 6.0,      # Immerkaer sigma above which denoise runs
    "contrast_min": 120.0,       # p95-p5 grey range below which CLAHE runs
    "blur_laplacian_min": 20.0,  # recorded as a quality flag only (no sharpening)
    "illumination_uneven_max": 20.0,  # background p95-p5 grey levels above which flat-field runs
    "quad_min_area": 0.25,
    "quad_max_area": 0.97,
    "quad_min_bg_contrast": 0.6,
}


@dataclass
class PreprocessResult:
    page: Page
    decisions: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def map_to_original(self, bbox) -> BBox:
        return map_to_original(self.page, bbox)


def map_to_original(page: Page, bbox) -> BBox:
    """Map a box on the processed page image back to original page coordinates."""
    return page.to_original(bbox)


def _params(cfg: Any) -> dict[str, Any]:
    raw = getattr(cfg, "raw", None) or {}
    user = raw.get("preprocess") if isinstance(raw.get("preprocess"), dict) else {}
    return {**DEFAULTS, **user}


def _metrics(img: np.ndarray) -> dict[str, float]:
    return {"laplacian_var": round(ops.laplacian_variance(img), 3),
            "noise_sigma": round(ops.noise_sigma(img), 3),
            "contrast": round(ops.contrast(img), 3)}


def preprocess_page(page: Page, *, params: dict[str, Any] | None = None,
                    orient_fn: Callable[[np.ndarray], int] | None = None,
                    trace: Any = None) -> PreprocessResult:
    """Return a new processed Page (the input page is untouched) plus the recorded decisions."""
    p = {**DEFAULTS, **(params or {})}
    if orient_fn is None:
        from radintake.tools.ocr import orientation

        orient_fn = orientation
    decisions: list[dict[str, Any]] = []

    def record(op: str, reason: str, **nums: float) -> None:
        d = {"op": op, "reason": reason, **{k: round(float(v), 4) for k, v in nums.items()}}
        decisions.append(d)
        if trace is not None:
            trace.event("preprocess_decision", page=page.index, **d)

    cur = page  # never mutate ``page``; ops return new arrays
    m0 = _metrics(cur.image)
    record("metrics", "input_quality", **m0)
    tl = getattr(page, "text_layer", None)
    if tl is not None and tl.usable and not tl.mixed:
        # born-digital page: text comes from the PDF itself, the raster needs no repair
        record("skip", "pdf_text_layer")
        return PreprocessResult(page=cur, decisions=decisions, metrics=m0)
    if m0["laplacian_var"] < p["blur_laplacian_min"]:
        record("flag_blur", "laplacian_var_below_min", laplacian_var=m0["laplacian_var"],
               threshold=p["blur_laplacian_min"])

    # 1. dewarp (photos: a page quad on a darker background) - before orientation, so the
    #    orientation check sees a flat page rather than a skewed photo
    quad = ops.find_page_quad(cur.image, min_area=p["quad_min_area"], max_area=p["quad_max_area"],
                              min_bg_contrast=p["quad_min_bg_contrast"])
    if quad is not None:
        img, t = ops.dewarp(cur.image, quad)
        cur = cur.with_image(img, t)
        record("dewarp", "page_quad_found", out_w=img.shape[1], out_h=img.shape[0])

    # 2. uneven lighting / shadows (photos): Tesseract binarises with one global threshold, so a
    #    lighting gradient turns half the page into noise. Photometric only - geometry unchanged.
    uneven = ops.illumination_unevenness(cur.image)
    if uneven > p["illumination_uneven_max"]:
        cur = cur.with_image(ops.flatten_illumination(cur.image), Transform.identity())
        record("flatten_illumination", "background_uneven", unevenness=uneven,
               threshold=p["illumination_uneven_max"])

    # 3. orientation
    rot = int(orient_fn(cur.image)) % 360
    if rot in (90, 180, 270):
        img, t = ops.rotate_cw(cur.image, rot)
        cur = cur.with_image(img, t)
        record("rotate", "orientation", degrees_cw=rot)
    else:
        record("keep_orientation", "orientation_upright", degrees_cw=0)

    # 3b. deskew
    ang = ops.skew_angle(cur.image, max_angle=p["deskew_max_deg"])
    if abs(ang) > p["deskew_min_deg"]:
        img, t = ops.deskew(cur.image, ang)
        cur = cur.with_image(img, t)
        record("deskew", "skew_above_min", skew_deg=ang, threshold=p["deskew_min_deg"])

    # 4. denoise / contrast only when the metrics demand it (geometry unchanged)
    m1 = _metrics(cur.image)
    img = cur.image
    if m1["noise_sigma"] > p["noise_sigma_max"]:
        img = ops.denoise(img)
        record("denoise", "noise_above_max", noise_sigma=m1["noise_sigma"], threshold=p["noise_sigma_max"])
    if m1["contrast"] < p["contrast_min"]:
        img = ops.clahe(img)
        record("clahe", "contrast_below_min", contrast=m1["contrast"], threshold=p["contrast_min"])
    if img is not cur.image:
        cur = cur.with_image(img, Transform.identity())
    return PreprocessResult(page=cur, decisions=decisions, metrics=m1)


@register_stage("preprocess", applies_to=DOCUMENT_TYPES, order=20, failure_flag="preprocess_failed")
def preprocess_stage(ctx: StageContext) -> dict[str, Any]:
    pages: list[Page] = ctx.data.get("pages") or []
    params = _params(ctx.config)
    out: list[Page] = []
    t0 = time.perf_counter()
    for pg in pages:
        res = preprocess_page(pg, params=params, trace=ctx.trace)
        out.append(res.page)
    ctx.trace.event("preprocess_done", n_pages=len(out),
                    duration_ms=round((time.perf_counter() - t0) * 1000.0, 3))
    return {"pages": out, "original_pages": pages}
