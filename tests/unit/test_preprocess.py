"""Preprocessing agent: random rotation/skew/perspective recovery and 2 px box round trip."""

from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from radintake.agents.preprocess import map_to_original, preprocess_page
from radintake.core.types import BBox, Page, Transform
from radintake.orchestrator.stages import registered

PW, PH = 700, 900
MARKERS = [(60, 60), (PW - 60, 60), (PW - 60, PH - 60), (60, PH - 60), (PW // 2, PH // 2 + 90)]
CW, CH = 1000, 1300
BG = 110


def _page_image() -> np.ndarray:
    img = Image.new("L", (PW, PH), 255)
    d = ImageDraw.Draw(img)
    for i, y in enumerate(range(120, PH - 120, 38)):  # text-like lines of varying length
        d.rectangle((90, y, PW - 90 - (i % 4) * 70, y + 12), fill=30)
    for x, y in MARKERS:
        d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=0)
    return np.asarray(img)


def _degrade(rng: np.random.Generator, *, photo: bool):
    """Returns (degraded image, forward map of page coords -> degraded coords, k quarter turns)."""
    page = _page_image()
    skew = float(rng.uniform(-8, 8))
    k = int(rng.integers(0, 4))
    if photo:
        canvas = np.full((CH, CW), BG, np.uint8)
        off = np.array([(CW - PW) / 2.0, (CH - PH) / 2.0])
        src = np.array([[0, 0], [PW, 0], [PW, PH], [0, PH]], np.float64)
        jit = rng.uniform(-0.04, 0.04, (4, 2)) * np.array([PW, PH])
        dst = src + off + jit
        rot = Transform.rotation(skew, CW / 2.0, CH / 2.0)
        persp = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)
        g = Transform(persp).then(rot)
        warped = cv2.warpPerspective(page, g.matrix, (CW, CH), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
        base = canvas.copy()
        mask = cv2.warpPerspective(np.full_like(page, 255), g.matrix, (CW, CH), flags=cv2.INTER_NEAREST)
        base[mask > 0] = warped[mask > 0]
        base = np.clip(base.astype(np.float32) + rng.normal(0, 2, base.shape), 0, 255).astype(np.uint8)
        w, h = CW, CH
    else:
        pad = 80  # white margin so skewed markers stay inside the frame
        w, h = PW + 2 * pad, PH + 2 * pad
        g = Transform.translation(pad, pad).then(Transform.rotation(skew, w / 2.0, h / 2.0))
        base = cv2.warpPerspective(page, g.matrix, (w, h), borderValue=255)
    img = np.ascontiguousarray(np.rot90(base, k))
    g = g.then(Transform.rot90(k, w, h))
    return img, g, k


def _centroid(img: np.ndarray, x: float, y: float, r: int = 16) -> tuple[float, float] | None:
    x0, y0 = round(x) - r, round(y) - r
    if x0 < 0 or y0 < 0 or x0 + 2 * r >= img.shape[1] or y0 + 2 * r >= img.shape[0]:
        return None
    win = img[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1].astype(np.float64)
    wgt = np.clip(60.0 - win, 0, None)
    if wgt.sum() < 50:
        return None
    ys, xs = np.mgrid[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1]
    return float((wgt * xs).sum() / wgt.sum()), float((wgt * ys).sum() / wgt.sum())


@pytest.mark.parametrize("photo", [True, False])
def test_random_compositions_recover_upright_and_round_trip(photo):
    rng = np.random.default_rng(140 + int(photo))
    n = 60 if photo else 50  # >= 100 random transforms across both modes
    for trial in range(n):
        img, g, k = _degrade(rng, photo=photo)
        before = img.copy()
        pg = Page(index=1, image=img, original_size=(img.shape[1], img.shape[0]))
        res = preprocess_page(pg, orient_fn=lambda _a, k=k: (90 * k) % 360)
        out = res.page
        assert np.array_equal(img, before), "input array must not be mutated"
        ops_done = {d["op"] for d in res.decisions}
        if photo:
            assert "dewarp" in ops_done, (trial, ops_done)
        for mx, my in MARKERS:
            # location of the marker in the degraded ("original") image, from the ground-truth warp
            ox, oy = g.map_points([(mx, my)])[0]
            # location in the processed image, via the page transform (detect, don't trust)
            px, py = out.transform.map_points([(ox, oy)])[0]
            c = _centroid(out.image, px, py)
            assert c is not None, (trial, "marker not at transformed location")
            # box round trip: a box around the detected marker maps back within 2 px
            back = map_to_original(out, BBox(c[0] - 3, c[1] - 3, c[0] + 3, c[1] + 3))
            bc = ((back.x0 + back.x1) / 2.0, (back.y0 + back.y1) / 2.0)
            assert abs(bc[0] - ox) <= 2.0 and abs(bc[1] - oy) <= 2.0, (trial, bc, (ox, oy))
            # upright: processed marker sits where it does on the upright page (1.5% of the page)
            # (the page's true size is unknown to a dewarper, so compare as fractions of the page)
            fx, fy = c[0] / out.size[0], c[1] / out.size[1]
            if photo:
                assert abs(fx - mx / PW) <= 0.015 and abs(fy - my / PH) <= 0.015, (trial, (fx, fy), (mx, my))
            else:
                assert abs(c[0] - (mx + 80)) <= 8 and abs(c[1] - (my + 80)) <= 8, (trial, c, (mx, my))


def test_source_file_unchanged_and_photo_upright(tmp_path):
    from radintake.tools.loader import load_pages
    from radintake.tools.ocr import orientation, tool_version
    from tests.integration.fixtures import build_corpus

    try:
        tool_version()
    except Exception:  # noqa: BLE001
        pytest.skip("tesseract not installed")
    build_corpus(tmp_path)
    src = tmp_path / "report_photo_rot90.jpg"
    h0 = hashlib.sha256(src.read_bytes()).hexdigest()
    pages = load_pages(src, "photo")
    res = preprocess_page(pages[0])
    assert hashlib.sha256(src.read_bytes()).hexdigest() == h0
    assert orientation(res.page.image) == 0
    assert any(d["op"] == "rotate" and d["degrees_cw"] in (90, 270) for d in res.decisions)
    # a processed-page box maps back inside the original page bounds
    b = map_to_original(res.page, BBox(100, 100, 300, 160))
    w, h = res.page.original_size
    assert 0 <= b.x0 < b.x1 <= w and 0 <= b.y0 < b.y1 <= h


def test_decisions_recorded_in_trace_with_metrics(tmp_path):
    from radintake.tracing.writer import TraceWriter

    case = TraceWriter(tmp_path / "tr").start_case("c1", "0" * 64)
    pg = Page(index=1, image=_page_image(), original_size=(PW, PH))
    res = preprocess_page(pg, orient_fn=lambda _a: 0, trace=case)
    lines = [json.loads(x) for x in case.path.read_text().splitlines()]
    dec = [r["data"] for r in lines if r["kind"] == "preprocess_decision"]
    assert [d["op"] for d in dec] == [d["op"] for d in res.decisions]
    metrics = next(d for d in dec if d["op"] == "metrics")
    assert {"laplacian_var", "noise_sigma", "contrast"} <= set(metrics)
    assert all(isinstance(v, (int, float)) for k, v in metrics.items() if k not in ("op", "reason", "page"))


def test_low_contrast_and_noise_trigger_photometric_ops():
    rng = np.random.default_rng(1)
    flat = (_page_image().astype(np.float32) * 0.2 + 140).astype(np.uint8)  # washed out
    noisy = np.clip(flat + rng.normal(0, 14, flat.shape), 0, 255).astype(np.uint8)
    res = preprocess_page(Page(index=1, image=noisy, original_size=(PW, PH)), orient_fn=lambda _a: 0)
    ops_done = [d["op"] for d in res.decisions]
    assert "denoise" in ops_done and "clahe" in ops_done
    clean = preprocess_page(Page(index=1, image=_page_image(), original_size=(PW, PH)), orient_fn=lambda _a: 0)
    assert not {"denoise", "clahe", "dewarp", "rotate"} & {d["op"] for d in clean.decisions}


def test_stage_registered():
    import radintake.agents.preprocess  # noqa: F401

    st = registered()["preprocess"]
    assert {"scan", "photo", "pdf", "tiff"} <= set(st.applies_to) and st.failure_flag == "preprocess_failed"
