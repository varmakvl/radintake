"""Degradation library on synthetic pages (no licensed data): GT box transform, determinism, writers."""

import io

import cv2
import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image

from data_prep.degrade import (
    TIERS,
    apply_homography,
    degrade_document,
    degrade_page,
    load_tiers,
    write_page,
    write_pdf,
    write_tiff,
)

CFG = load_tiers()
PAGE_W, PAGE_H = 640, 820
RECTS = [(60, 70, 260, 130), (380, 90, 560, 300), (100, 420, 330, 470), (420, 600, 590, 760)]
DARK = 128


def _page(rects=RECTS, color=False):
    img = np.full((PAGE_H, PAGE_W), 255, np.uint8)
    for x0, y0, x1, y1 in rects:
        cv2.rectangle(img, (x0, y0), (x1, y1), 0, thickness=-1)
    return np.stack([img] * 3, axis=-1) if color else img


def _dark_bbox(img):
    gray = img if img.ndim == 2 else img.mean(axis=2)
    ys, xs = np.nonzero(gray < DARK)
    assert len(xs), "rectangle vanished from the degraded page"
    return xs, ys, (xs.min(), ys.min(), xs.max(), ys.max())


def _iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0 + 1) * max(0.0, iy1 - iy0 + 1)
    area = lambda r: (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
    return inter / (area(a) + area(b) - inter)


def _check_boxes(tier, seed, overrides=None):
    """Each rectangle is degraded alone with the same seed (same H), so its dark pixels are its own."""
    full = degrade_page(_page(), RECTS, tier, seed, overrides=overrides, geometry_only=True)
    for i, rect in enumerate(RECTS):
        alone = degrade_page(_page([rect]), [rect], tier, seed, overrides=overrides, geometry_only=True)
        np.testing.assert_array_equal(alone.H, full.H)
        x0, y0, x1, y1 = full.boxes[i].aabb
        xs, ys, dark_box = _dark_bbox(alone.image)
        inside = (xs >= x0 - 0.5) & (xs <= x1 + 0.5) & (ys >= y0 - 0.5) & (ys <= y1 + 0.5)
        coverage = inside.mean()
        iou = _iou(full.boxes[i].aabb, dark_box)
        assert coverage >= 0.9, (tier, seed, overrides, i, coverage)
        assert iou >= 0.7, (tier, seed, overrides, i, iou)
    return full


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("quarter", [0, 90, 180, 270])
def test_transformed_boxes_cover_rectangles(tier, quarter):
    for seed in (140, 141, 142):
        res = _check_boxes(tier, seed, {"rotate90": quarter})
        assert res.ops[0] == {"op": "rotate90", "deg": quarter}


@pytest.mark.parametrize("tier", TIERS)
def test_tier_defaults_cover_rectangles(tier):
    for seed in range(140, 150):
        _check_boxes(tier, seed)


def test_strong_perspective_and_skew_cover_rectangles():
    over = {"perspective": 0.1, "skew_deg": [-6, 6], "rotation_deg": [-10, 10], "rotate90": 180}
    res = _check_boxes("phone_photo", 140, over)
    assert any(abs(v) > 0.01 for row in res.ops[4]["corner_jitter"] for v in row)


def test_exact_quarter_turns_are_pixel_exact():
    img = _page()
    for quarter, k in ((90, -1), (180, 2), (270, 1)):
        over = {
            "rotate90": quarter,
            "scale": [1, 1],
            "rotation_deg": [0, 0],
            "skew_deg": [0, 0],
            "perspective": 0.0,
            "margin_px": [0, 0],
        }
        res = degrade_page(img, RECTS, "clean_scan", 140, overrides=over, geometry_only=True)
        np.testing.assert_array_equal(res.image, np.rot90(img, k))


def test_inverse_h_maps_box_corners_back():
    for tier in TIERS:
        for seed in (140, 7, 99):
            res = degrade_page(_page(), RECTS, tier, seed, overrides={"rotate90": 90 * (seed % 4)})
            for rect, box in zip(RECTS, res.boxes, strict=True):
                x0, y0, x1, y1 = rect
                orig = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
                back = apply_homography(np.linalg.inv(res.H), box.quad)
                assert np.abs(back - orig).max() <= 2.0
                assert np.abs(res.unwarp_points(box.quad) - orig).max() <= 2.0


def test_same_seed_identical_different_seed_differs(tmp_path):
    for tier in TIERS:
        a = degrade_page(_page(color=True), RECTS, tier, 140)
        b = degrade_page(_page(color=True), RECTS, tier, 140)
        c = degrade_page(_page(color=True), RECTS, tier, 141)
        assert a.image.tobytes() == b.image.tobytes()
        assert a.manifest_record() == b.manifest_record()
        assert a.image.shape != c.image.shape or a.image.tobytes() != c.image.tobytes()
        assert not np.allclose(a.H, c.H)
        pa = write_page(a.image, tmp_path / f"{tier}_a.png")
        pb = write_page(b.image, tmp_path / f"{tier}_b.png")
        assert pa.read_bytes() == pb.read_bytes()

    pages = [_page(), _page(RECTS[:2])]
    d1 = degrade_document(pages, [RECTS, RECTS[:2]], "poor_scan", 140)
    d2 = degrade_document(pages, [RECTS, RECTS[:2]], "poor_scan", 140)
    d3 = degrade_document(pages, [RECTS, RECTS[:2]], "poor_scan", 141)
    assert [r.seed for r in d1] == [r.seed for r in d2] != [r.seed for r in d3]
    assert d1[0].seed != d1[1].seed
    f1 = write_pdf([r.image for r in d1], tmp_path / "d1.pdf")
    f2 = write_pdf([r.image for r in d2], tmp_path / "d2.pdf")
    f3 = write_pdf([r.image for r in d3], tmp_path / "d3.pdf")
    assert f1.read_bytes() == f2.read_bytes() != f3.read_bytes()
    t1 = write_tiff([r.image for r in d1], tmp_path / "d1.tif")
    t2 = write_tiff([r.image for r in d2], tmp_path / "d2.tif")
    assert t1.read_bytes() == t2.read_bytes()


def test_default_seed_is_base_seed():
    assert CFG["base_seed"] == 140
    assert degrade_page(_page(), RECTS, "poor_scan").seed == 140


def test_geometry_only_shares_h_with_full_run():
    for tier in TIERS:
        full = degrade_page(_page(), RECTS, tier, 140)
        geo = degrade_page(_page(), RECTS, tier, 140, geometry_only=True)
        np.testing.assert_array_equal(full.H, geo.H)
        assert full.image.shape == geo.image.shape


def test_full_tier_ops_recorded_and_boxes_in_bounds():
    over = {"stamp_prob": 1.0, "lines_prob": 1.0}
    for tier in TIERS:
        for img in (_page(), _page(color=True)):
            res = degrade_page(img, RECTS, tier, 140, overrides=over)
            w, h = res.size
            assert res.image.shape[:2] == (h, w) and res.image.ndim == img.ndim
            assert res.image.dtype == np.uint8
            names = [op["op"] for op in res.ops]
            for name in (
                "rotate90",
                "scale",
                "rotate",
                "skew",
                "perspective",
                "fit",
                "lighting",
                "stamp",
                "lines",
                "blur",
                "noise",
                "jpeg",
            ):
                assert name in names, (tier, name)
            for box in res.boxes:
                x0, y0, x1, y1 = box.aabb
                assert 0 <= x0 <= x1 <= w - 1 and 0 <= y0 <= y1 <= h - 1
            rec = res.manifest_record()
            assert rec["seed"] == 140 and rec["tier"] == tier and len(rec["boxes"]) == len(RECTS)


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        degrade_page(_page(), RECTS, "fax")


@pytest.mark.parametrize("n_pages", [1, 3])
def test_multipage_writers_page_count(tmp_path, n_pages):
    pages = [r.image for r in degrade_document([_page()] * n_pages, None, "phone_photo", 140)]
    pdf = write_pdf(pages, tmp_path / "doc.pdf")
    doc = pdfium.PdfDocument(str(pdf))
    try:
        assert len(doc) == n_pages
    finally:
        doc.close()
    tif = write_tiff(pages, tmp_path / "doc.tif")
    with Image.open(tif) as im:
        assert im.n_frames == n_pages
        for i in range(n_pages):
            im.seek(i)
            assert im.size == (pages[i].shape[1], pages[i].shape[0])


def test_single_page_writers_roundtrip(tmp_path):
    img = degrade_page(_page(color=True), RECTS, "clean_scan", 140).image
    png = write_page(img, tmp_path / "p.png")
    np.testing.assert_array_equal(np.asarray(Image.open(png)), img)
    jpg = write_page(img, tmp_path / "p.jpg", jpeg_quality=80)
    with Image.open(io.BytesIO(jpg.read_bytes())) as im:
        assert im.format == "JPEG" and im.size == (img.shape[1], img.shape[0])
    with pytest.raises(ValueError):
        write_page(img, tmp_path / "p.bmp")
