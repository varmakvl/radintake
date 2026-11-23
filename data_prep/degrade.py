"""Seeded page degradation: clean_scan / poor_scan / phone_photo tiers with GT box transform.

Geometry. Every geometric op (quarter-turn rotation, small rotation, skew, low-DPI scale,
perspective warp, background margin) is a 3x3 homography. They are composed into ONE matrix
``H`` per page, the page is warped once with it, and each ground-truth box is pushed through
the same ``H``: as a quad (its four corners, exact under a homography) and as the axis-aligned
box around that quad, clipped to the degraded image. ``np.linalg.inv(H)`` maps degraded
coordinates back to the original page, so evaluation can go either way.

Coordinates follow OpenCV: integer (x, y) is a pixel centre, x to the right, y down.
Boxes are ``(x0, y0, x1, y1)`` in original-page pixels.

Photometry (uneven lighting / shadow, stamps, ruled lines, blur, gaussian noise, JPEG
round-trip) runs after the warp, in degraded coordinates, and never moves content.

Randomness. Every sampled value comes from ``np.random.default_rng(seed)`` (``seed``
defaults to the config ``base_seed``, 140) and is returned in ``DegradeResult.ops`` along
with the seed, so a manifest can record and replay it. All parameters are sampled before
anything is applied, so ``geometry_only=True`` gives the same ``H`` as the full run.

Images are uint8 arrays, either ``(h, w)`` grayscale or ``(h, w, 3)`` RGB.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

TIERS = ("clean_scan", "poor_scan", "phone_photo")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "degrade_tiers.yaml"
DEFAULT_BASE_SEED = 140

Box = Sequence[float]


# --------------------------------------------------------------------------- config


def load_tiers(path: str | Path | None = None) -> dict[str, Any]:
    """Load the tier config (``configs/degrade_tiers.yaml`` by default)."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    missing = [t for t in TIERS if t not in cfg.get("tiers", {})]
    if missing:
        raise ValueError(f"degrade config lacks tiers: {missing}")
    cfg.setdefault("base_seed", DEFAULT_BASE_SEED)
    return cfg


def _tier_params(cfg: Mapping[str, Any], tier: str, overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if tier not in cfg["tiers"]:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(cfg['tiers'])}")
    params = dict(cfg["tiers"][tier])
    for key, value in (overrides or {}).items():
        if key == "rotate90" and isinstance(value, int | float):
            value = {int(value): 1.0}  # force an exact quarter turn
        params[key] = value
    return params


# --------------------------------------------------------------------------- results


@dataclass
class TransformedBox:
    """A GT box in degraded-image coordinates."""

    quad: np.ndarray  # (4, 2) float: corners (x0,y0) (x1,y0) (x1,y1) (x0,y1) mapped through H
    aabb: tuple[float, float, float, float]  # axis-aligned box around quad, clipped to the image

    def to_dict(self) -> dict[str, Any]:
        return {
            "quad": [[round(float(x), 3), round(float(y), 3)] for x, y in self.quad],
            "aabb": [round(float(v), 3) for v in self.aabb],
        }


@dataclass
class DegradeResult:
    image: np.ndarray
    H: np.ndarray  # 3x3 float64: original page pixel -> degraded pixel
    boxes: list[TransformedBox]
    ops: list[dict[str, Any]]
    seed: int
    tier: str
    size: tuple[int, int]  # (width, height) of the degraded image
    geometry_only: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def manifest_record(self) -> dict[str, Any]:
        """JSON-serialisable record of everything needed to reproduce / invert this page."""
        return {
            "tier": self.tier,
            "seed": int(self.seed),
            "geometry_only": self.geometry_only,
            "size": [int(self.size[0]), int(self.size[1])],
            "H": [[float(v) for v in row] for row in self.H],
            "ops": self.ops,
            "boxes": [b.to_dict() for b in self.boxes],
        }

    def unwarp_points(self, pts: np.ndarray) -> np.ndarray:
        """Map (N, 2) degraded-image points back to original-page coordinates."""
        return apply_homography(np.linalg.inv(self.H), pts)


# --------------------------------------------------------------------------- geometry


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N, 2) points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H, dtype=np.float64).T
    return homog[:, :2] / homog[:, 2:3]


def _translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def _rot90(quarter_deg: int) -> np.ndarray:
    k = (quarter_deg // 90) % 4
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]  # exact cos/sin of k*90 (clockwise on screen)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotate(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _shear(deg: float) -> np.ndarray:
    return np.array([[1.0, math.tan(math.radians(deg)), 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _scale(s: float) -> np.ndarray:
    return np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]])


def page_corners(width: int, height: int) -> np.ndarray:
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float64)


def transform_boxes(boxes: Sequence[Box], H: np.ndarray, size: tuple[int, int]) -> list[TransformedBox]:
    """Transform ``(x0, y0, x1, y1)`` boxes through ``H`` into an image of ``size`` (w, h)."""
    w, h = size
    out: list[TransformedBox] = []
    for x0, y0, x1, y1 in boxes:
        quad = apply_homography(H, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64))
        xs, ys = quad[:, 0], quad[:, 1]
        aabb = (
            float(np.clip(xs.min(), 0, w - 1)),
            float(np.clip(ys.min(), 0, h - 1)),
            float(np.clip(xs.max(), 0, w - 1)),
            float(np.clip(ys.max(), 0, h - 1)),
        )
        out.append(TransformedBox(quad=quad, aabb=aabb))
    return out


# --------------------------------------------------------------------------- sampling


def _uniform(rng: np.random.Generator, rng_spec: Sequence[float]) -> float:
    lo, hi = float(rng_spec[0]), float(rng_spec[1])
    return lo if hi <= lo else float(rng.uniform(lo, hi))


def _sample_params(rng: np.random.Generator, p: Mapping[str, Any], width: int, height: int) -> dict[str, Any]:
    """Sample every op parameter in a fixed order (geometry first, then photometry)."""
    quarters = sorted((int(k), float(v)) for k, v in p["rotate90"].items())
    probs = np.array([v for _, v in quarters], dtype=np.float64)
    q_idx = int(rng.choice(len(quarters), p=probs / probs.sum()))
    s: dict[str, Any] = {
        "rotate90": quarters[q_idx][0],
        "scale": _uniform(rng, p["scale"]),
        "rotation_deg": _uniform(rng, p["rotation_deg"]),
        "skew_deg": _uniform(rng, p["skew_deg"]),
        # perspective: per-corner (dx, dy) as a fraction of min(width, height)
        "perspective": (rng.uniform(-1.0, 1.0, size=(4, 2)) * float(p["perspective"])).tolist(),
        "margin_px": round(_uniform(rng, p["margin_px"])),
        "background": round(_uniform(rng, p["background"])),
    }
    s["lighting"] = {"strength": _uniform(rng, p["lighting"]), "angle_deg": float(rng.uniform(0, 360))}
    s["shadow"] = {
        "strength": _uniform(rng, p["shadow"]),
        "angle_deg": float(rng.uniform(0, 360)),
        "offset": float(rng.uniform(-0.3, 0.3)),
        "softness": float(rng.uniform(0.03, 0.12)),
    }
    s["stamp"] = None
    if rng.uniform() < float(p["stamp_prob"]):
        s["stamp"] = {
            "cx": float(rng.uniform(0.15, 0.85)),
            "cy": float(rng.uniform(0.15, 0.85)),
            "radius": float(rng.uniform(0.05, 0.1)),
            "shape": ["circle", "rect"][int(rng.integers(0, 2))],
            "color": [[200, 30, 40], [30, 50, 190], [120, 30, 150]][int(rng.integers(0, 3))],
            "alpha": float(rng.uniform(0.4, 0.75)),
            "text": ["RECEIVED", "COPY", "VERIFIED", "SCANNED"][int(rng.integers(0, 4))],
        }
    s["lines"] = []
    if rng.uniform() < float(p["lines_prob"]):
        for _ in range(int(rng.integers(1, 4))):
            s["lines"].append(
                {
                    "orientation": ["h", "v"][int(rng.integers(0, 2))],
                    "pos": float(rng.uniform(0.02, 0.98)),
                    "value": int(rng.integers(0, 140)),
                    "thickness": int(rng.integers(1, 3)),
                }
            )
    s["blur_sigma"] = _uniform(rng, p["blur_sigma"])
    s["noise_sigma"] = _uniform(rng, p["noise_sigma"])
    s["jpeg_quality"] = round(_uniform(rng, p["jpeg_quality"]))
    # noise realisation is drawn later from a child seed so its size never shifts other params
    s["noise_seed"] = int(rng.integers(0, 2**31 - 1))
    return s


def _compose_geometry(s: Mapping[str, Any], width: int, height: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Compose the sampled geometric ops into one homography and the output canvas size."""
    corners = page_corners(width, height)
    A = _shear(s["skew_deg"]) @ _rotate(s["rotation_deg"]) @ _scale(s["scale"]) @ _rot90(s["rotate90"])
    mapped = apply_homography(A, corners)
    span = min(np.ptp(mapped[:, 0]), np.ptp(mapped[:, 1]))
    jittered = mapped + np.asarray(s["perspective"], dtype=np.float64) * span
    P = cv2.getPerspectiveTransform(mapped.astype(np.float32), jittered.astype(np.float32)).astype(np.float64)
    G = P @ A
    final = apply_homography(G, corners)
    m = s["margin_px"]
    # Snap the fit translation so exact quarter turns stay on the integer pixel grid.
    min_x, min_y = np.floor(final.min(axis=0) + 1e-6)
    H = _translate(-min_x + m, -min_y + m) @ G
    final = apply_homography(H, corners)
    out_w = math.ceil(final[:, 0].max() - 1e-6) + 1 + m
    out_h = math.ceil(final[:, 1].max() - 1e-6) + 1 + m
    H = H / H[2, 2]
    return H, (out_w, out_h)


# --------------------------------------------------------------------------- photometry


def _apply_lighting(img: np.ndarray, light: Mapping[str, Any], shadow: Mapping[str, Any]) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = xx / max(w - 1, 1) - 0.5, yy / max(h - 1, 1) - 0.5
    factor = np.ones((h, w), dtype=np.float32)
    if light["strength"] > 0:
        a = math.radians(light["angle_deg"])
        proj = nx * math.cos(a) + ny * math.sin(a)
        g = (proj - proj.min()) / max(float(np.ptp(proj)), 1e-6)
        factor *= 1.0 - float(light["strength"]) * g
    if shadow["strength"] > 0:
        a = math.radians(shadow["angle_deg"])
        d = nx * math.cos(a) + ny * math.sin(a) - float(shadow["offset"])
        soft = 1.0 / (1.0 + np.exp(-d / float(shadow["softness"])))
        factor *= 1.0 - float(shadow["strength"]) * soft
    if img.ndim == 3:
        factor = factor[..., None]
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _apply_stamp(img: np.ndarray, st: Mapping[str, Any]) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()
    color = tuple(int(c) for c in st["color"]) if img.ndim == 3 else int(np.mean(st["color"]))
    cx, cy = int(st["cx"] * w), int(st["cy"] * h)
    r = max(int(st["radius"] * min(w, h)), 8)
    thick = max(r // 12, 2)
    if st["shape"] == "circle":
        cv2.circle(overlay, (cx, cy), r, color, thick, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), int(r * 0.8), color, max(thick // 2, 1), cv2.LINE_AA)
    else:
        cv2.rectangle(
            overlay, (cx - int(r * 1.6), cy - r // 2), (cx + int(r * 1.6), cy + r // 2), color, thick
        )
    font_scale = max(r / 60.0, 0.3)
    (tw, th), _ = cv2.getTextSize(st["text"], cv2.FONT_HERSHEY_DUPLEX, font_scale, max(thick // 2, 1))
    cv2.putText(
        overlay,
        st["text"],
        (cx - tw // 2, cy + th // 2),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        color,
        max(thick // 2, 1),
        cv2.LINE_AA,
    )
    a = float(st["alpha"])
    return cv2.addWeighted(overlay, a, img, 1.0 - a, 0.0)


def _apply_lines(img: np.ndarray, lines: Sequence[Mapping[str, Any]]) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for ln in lines:
        color = (ln["value"],) * 3 if img.ndim == 3 else int(ln["value"])
        if ln["orientation"] == "h":
            y = int(ln["pos"] * (h - 1))
            cv2.line(out, (0, y), (w - 1, y), color, int(ln["thickness"]))
        else:
            x = int(ln["pos"] * (w - 1))
            cv2.line(out, (x, 0), (x, h - 1), color, int(ln["thickness"]))
    return out


def _jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB" if img.ndim == 3 else "L"))


# --------------------------------------------------------------------------- public API


def degrade_page(
    image: np.ndarray,
    boxes: Sequence[Box] = (),
    tier: str = "clean_scan",
    seed: int | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    geometry_only: bool = False,
) -> DegradeResult:
    """Degrade one page and transform its GT boxes.

    ``overrides`` replaces tier keys (e.g. ``{"rotate90": 180}`` forces an exact half turn).
    ``geometry_only`` applies the same homography but skips photometric ops and fills the
    border white (a noise-free view of the geometry for testing / QA).
    """
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("image must be a uint8 (h, w) or (h, w, 3) array")
    cfg = config or load_tiers()
    seed = int(cfg.get("base_seed", DEFAULT_BASE_SEED) if seed is None else seed)
    params = _tier_params(cfg, tier, overrides)
    h, w = image.shape[:2]
    rng = np.random.default_rng(seed)
    s = _sample_params(rng, params, w, h)
    H, size = _compose_geometry(s, w, h)

    bg = 255 if geometry_only else s["background"]
    border = (bg, bg, bg) if image.ndim == 3 else bg
    out = cv2.warpPerspective(
        image, H, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border
    )

    ops: list[dict[str, Any]] = [
        {"op": "rotate90", "deg": s["rotate90"]},
        {"op": "scale", "factor": s["scale"]},
        {"op": "rotate", "deg": s["rotation_deg"]},
        {"op": "skew", "deg": s["skew_deg"]},
        {"op": "perspective", "corner_jitter": s["perspective"]},
        {"op": "fit", "margin_px": s["margin_px"], "background": bg},
    ]
    if not geometry_only:
        out = _apply_lighting(out, s["lighting"], s["shadow"])
        ops.append({"op": "lighting", **s["lighting"]})
        ops.append({"op": "shadow", **s["shadow"]})
        if s["stamp"]:
            out = _apply_stamp(out, s["stamp"])
            ops.append({"op": "stamp", **s["stamp"]})
        if s["lines"]:
            out = _apply_lines(out, s["lines"])
            ops.append({"op": "lines", "lines": s["lines"]})
        if s["blur_sigma"] > 0:
            out = cv2.GaussianBlur(out, (0, 0), s["blur_sigma"])
        ops.append({"op": "blur", "sigma": s["blur_sigma"]})
        if s["noise_sigma"] > 0:
            noise = np.random.default_rng(s["noise_seed"]).normal(0.0, s["noise_sigma"], out.shape)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        ops.append({"op": "noise", "sigma": s["noise_sigma"], "seed": s["noise_seed"]})
        out = _jpeg_roundtrip(out, s["jpeg_quality"])
        ops.append({"op": "jpeg", "quality": s["jpeg_quality"]})

    return DegradeResult(
        image=np.ascontiguousarray(out),
        H=H,
        boxes=transform_boxes(boxes, H, size),
        ops=ops,
        seed=seed,
        tier=tier,
        size=size,
        geometry_only=geometry_only,
    )


def page_seeds(seed: int, n_pages: int) -> list[int]:
    """Independent, reproducible per-page seeds derived from a document seed."""
    return [int(ss.generate_state(1)[0]) for ss in np.random.SeedSequence(int(seed)).spawn(n_pages)]


def degrade_document(
    pages: Sequence[np.ndarray],
    boxes_per_page: Sequence[Sequence[Box]] | None = None,
    tier: str = "clean_scan",
    seed: int | None = None,
    **kwargs: Any,
) -> list[DegradeResult]:
    """Degrade every page of a document with the same tier and per-page derived seeds."""
    cfg = kwargs.pop("config", None) or load_tiers()
    seed = int(cfg.get("base_seed", DEFAULT_BASE_SEED) if seed is None else seed)
    boxes_per_page = boxes_per_page or [()] * len(pages)
    if len(boxes_per_page) != len(pages):
        raise ValueError("boxes_per_page must have one entry per page")
    return [
        degrade_page(img, bxs, tier, s, config=cfg, **kwargs)
        for img, bxs, s in zip(pages, boxes_per_page, page_seeds(seed, len(pages)), strict=True)
    ]


# --------------------------------------------------------------------------- writers


def _to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(img))  # (h, w) -> "L", (h, w, 3) -> "RGB"


def write_page(img: np.ndarray, path: str | Path, jpeg_quality: int = 92, dpi: int = 200) -> Path:
    """Write one page as PNG or JPEG (chosen by suffix)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".png":
        _to_pil(img).save(path, format="PNG", dpi=(dpi, dpi))
    elif suffix in (".jpg", ".jpeg"):
        _to_pil(img).save(path, format="JPEG", quality=int(jpeg_quality), dpi=(dpi, dpi))
    else:
        raise ValueError(f"unsupported single-page format: {suffix}")
    return path


def write_pdf(pages: Sequence[np.ndarray], path: str | Path, dpi: int = 200) -> Path:
    """Write a multi-page PDF (one image per page, lossless PNG streams, no timestamps)."""
    import img2pdf

    if not pages:
        raise ValueError("no pages to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blobs = []
    for img in pages:
        buf = io.BytesIO()
        _to_pil(img).save(buf, format="PNG")
        blobs.append(buf.getvalue())
    layout = img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))
    path.write_bytes(img2pdf.convert(blobs, layout_fun=layout, nodate=True))
    return path


def write_tiff(pages: Sequence[np.ndarray], path: str | Path, dpi: int = 200) -> Path:
    """Write a multi-page TIFF (deflate compressed)."""
    if not pages:
        raise ValueError("no pages to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = (_to_pil(p) for p in pages)
    first.save(
        path, format="TIFF", save_all=True, append_images=rest, compression="tiff_deflate", dpi=(dpi, dpi)
    )
    return path
