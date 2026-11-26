"""Phone-photo page detection (fallback line detector) and the coordinate round trip.

Synthetic images only (numpy / cv2 / PIL, nothing under data/): a rendered text page is warped by a
KNOWN homography onto a desk of similar brightness, then a lighting gradient and a soft shadow are
laid over the whole frame, so no global threshold separates paper from desk and the contour detector
(``find_page_quad``) misses the page. The fallback (``find_page_quad_lines``) must find it.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from radintake.agents.preprocess import map_to_original, preprocess_page
from radintake.core.types import BBox, Page, Transform
from radintake.tools import image_ops as ops

PW, PH = 760, 980          # upright page
CW, CH = 1100, 1400        # photo frame
MARKERS = [(50, 50), (PW - 50, 50), (PW - 50, PH - 50), (50, PH - 50)]
WORDS = ["Patient", "Name:", "Accession", "Findings:", "Impression:", "pleural", "effusion", "normal"]


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _text_page() -> tuple[np.ndarray, list[BBox]]:
    """Rendered text page and the page-coordinate box of every drawn word."""
    img = Image.new("L", (PW, PH), 250)
    d = ImageDraw.Draw(img)
    f = _font(22)
    boxes: list[BBox] = []
    y = 110
    for row in range(16):
        x = 90
        for c in range(4):
            w = WORDS[(row * 3 + c) % len(WORDS)]
            x0, y0, x1, y1 = d.textbbox((x, y), w, font=f)
            if x1 > PW - 80:
                break
            d.text((x, y), w, fill=20, font=f)
            boxes.append(BBox(x0, y0, x1, y1))
            x = x1 + 24
        y += 46
    d.rectangle((80, PH - 220, PW - 80, PH - 140), outline=40, width=2)  # a ruled table cell
    for x, y in MARKERS:
        d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=0)
    return np.asarray(img), boxes


def _photo(seed: int, *, bg: int = 145) -> tuple[np.ndarray, Transform, list[BBox]]:
    """(photo, known homography page -> photo, word boxes in page coords)."""
    rng = np.random.default_rng(seed)
    page, boxes = _text_page()
    src = np.array([[0, 0], [PW, 0], [PW, PH], [0, PH]], np.float64)
    dst = src + np.array([(CW - PW) / 2.0, (CH - PH) / 2.0]) + rng.uniform(-0.05, 0.05, (4, 2)) * [PW, PH]
    g = Transform(cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64))
    g = g.then(Transform.rotation(float(rng.uniform(-7, 7)), CW / 2.0, CH / 2.0))
    canvas = np.full((CH, CW), bg, np.uint8)
    cv2.warpPerspective(page, g.matrix, (CW, CH), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
    yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
    light = 1.0 - 0.45 * (xx / CW if seed % 2 else yy / CH)
    shadow = 1.0 - 0.3 / (1.0 + np.exp(-(xx - 0.55 * CW + 0.4 * yy) / 45.0))
    img = canvas.astype(np.float32) * light * shadow + rng.normal(0, 3, canvas.shape)
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0.8)
    return img, g, boxes


def _centroid(img: np.ndarray, x: float, y: float, r: int = 16) -> tuple[float, float] | None:
    x0, y0 = round(x) - r, round(y) - r
    if x0 < 0 or y0 < 0 or x0 + 2 * r >= img.shape[1] or y0 + 2 * r >= img.shape[0]:
        return None
    win = img[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1].astype(np.float64)
    wgt = np.clip(np.median(win) - 40.0 - win, 0, None)  # clearly darker than the local paper
    if wgt.sum() < 50:
        return None
    ys, xs = np.mgrid[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1]
    return float((wgt * xs).sum() / wgt.sum()), float((wgt * ys).sum() / wgt.sum())


@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_fallback_finds_low_contrast_shadowed_page(seed, tmp_path):
    from radintake.tracing.writer import TraceWriter

    img, g, _ = _photo(seed)
    contour = ops.find_page_quad(img)  # misses the page, or merges the bright desk into it
    assert contour is None or min(ops.quad_side_support(img, contour)) < 0.5, "contour detector must fail here"
    quad, detector, score, rejected = ops.detect_page_quad(img, {})
    assert detector == "lines" and quad is not None and score >= 0.5
    assert rejected == int(contour is not None)
    known = g.map_points([(0, 0), (PW, 0), (PW, PH), (0, PH)])
    err = np.linalg.norm(ops.order_quad(quad) - np.asarray(known), axis=1)
    assert err.max() <= 0.01 * np.hypot(PW, PH), (seed, err)  # corners within 1% of the diagonal (~12 px)

    case = TraceWriter(tmp_path / "tr").start_case("c1", "0" * 64)
    res = preprocess_page(Page(index=1, image=img, original_size=(CW, CH)), orient_fn=lambda _a: 0, trace=case)
    dec = [json.loads(x)["data"] for x in case.path.read_text().splitlines()
           if json.loads(x)["kind"] == "preprocess_decision"]
    dw = [d for d in dec if d["op"] == "dewarp"]
    assert dw and dw[0]["reason"] == "page_quad_fallback" and dw[0]["fallback"] == 1
    # detect, don't trust: every corner marker is found where the dewarped upright page has it
    for mx, my in MARKERS:
        ox, oy = g.map_points([(mx, my)])[0]
        px, py = res.page.transform.map_points([(ox, oy)])[0]
        c = _centroid(res.page.image, px, py)
        assert c is not None, (seed, (mx, my))
        assert abs(c[0] / res.page.size[0] - mx / PW) <= 0.015 and abs(c[1] / res.page.size[1] - my / PH) <= 0.015


@pytest.mark.parametrize("seed", [21, 22])
def test_word_boxes_round_trip_to_original_within_1_5px(seed):
    img, g, boxes = _photo(seed)
    res = preprocess_page(Page(index=1, image=img, original_size=(CW, CH)), orient_fn=lambda _a: 0)
    assert any(d["op"] == "dewarp" for d in res.decisions)
    t = res.page.transform
    for b in boxes:
        corners = [(b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1)]
        orig = np.asarray(g.map_points(corners))              # word corners in the photo (original)
        proc = np.asarray(t.map_points(orig))                 # the same word on the processed page
        # point round trip: every corner back within 1.5 px
        back = np.asarray(t.inverse().map_points(proc))
        assert np.abs(back - orig).max() <= 1.5, (seed, b)
        # box round trip: the processed-page box of the word maps back onto its original box
        pb = BBox(proc[:, 0].min(), proc[:, 1].min(), proc[:, 0].max(), proc[:, 1].max())
        ob = map_to_original(res.page, pb)
        known = BBox(orig[:, 0].min(), orig[:, 1].min(), orig[:, 0].max(), orig[:, 1].max())
        assert abs((ob.x0 + ob.x1) / 2 - (known.x0 + known.x1) / 2) <= 1.5, (seed, b, ob, known)
        assert abs((ob.y0 + ob.y1) / 2 - (known.y0 + known.y1) / 2) <= 1.5, (seed, b, ob, known)
        assert ob.x0 <= known.x0 + 1.5 and ob.y0 <= known.y0 + 1.5
        assert ob.x1 >= known.x1 - 1.5 and ob.y1 >= known.y1 - 1.5


def test_clean_scan_without_paper_edge_is_not_dewarped():
    page, _ = _text_page()
    framed = page.copy()
    cv2.rectangle(framed, (20, 20), (PW - 20, PH - 20), 30, 2)     # printed page frame, not a paper edge
    rng = np.random.default_rng(3)
    shaded = np.clip(page.astype(np.float32) * np.linspace(1.0, 0.8, PH)[:, None]
                     + rng.normal(0, 2, page.shape), 0, 255).astype(np.uint8)
    for im in (page, framed, shaded):
        assert ops.find_page_quad_lines(im)[0] is None
        res = preprocess_page(Page(index=1, image=im, original_size=(PW, PH)), orient_fn=lambda _a: 0)
        assert "dewarp" not in {d["op"] for d in res.decisions}
        assert np.allclose(res.page.transform.matrix, np.eye(3))


@pytest.mark.parametrize("k", [1, 2])
def test_rotated_photo_ends_upright_with_original_boxes(k):
    img, g, _ = _photo(31)
    rot_img = np.ascontiguousarray(np.rot90(img, k))  # photo taken turned by 90*k degrees
    g = g.then(Transform.rot90(k, CW, CH))
    res = preprocess_page(Page(index=1, image=rot_img, original_size=(rot_img.shape[1], rot_img.shape[0])),
                          orient_fn=lambda _a: 90 * k)
    ops_done = [d["op"] for d in res.decisions]
    assert "dewarp" in ops_done and "rotate" in ops_done, ops_done
    out = res.page
    for mx, my in MARKERS:
        ox, oy = g.map_points([(mx, my)])[0]
        px, py = out.transform.map_points([(ox, oy)])[0]
        c = _centroid(out.image, px, py)
        assert c is not None, (k, (mx, my))
        assert abs(c[0] / out.size[0] - mx / PW) <= 0.015 and abs(c[1] / out.size[1] - my / PH) <= 0.015
        back = map_to_original(out, BBox(c[0] - 3, c[1] - 3, c[0] + 3, c[1] + 3))
        assert abs((back.x0 + back.x1) / 2 - ox) <= 1.5 and abs((back.y0 + back.y1) / 2 - oy) <= 1.5
