"""Unit tests for the fixed input router and the page loader."""

from __future__ import annotations

import numpy as np
import pytest

from radintake.orchestrator.router import route, scan_or_photo, sniff
from radintake.tools.loader import load_pages
from tests.integration.fixtures import PAGE_H, PAGE_W, build_corpus


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    folder = tmp_path_factory.mktemp("router_corpus")
    build_corpus(folder)
    return folder


@pytest.mark.parametrize(("name", "allowed"), [
    ("report_clean.png", {"scan"}),
    ("report_photo_rot90.jpg", {"photo", "scan"}),
    ("report_injected_multipage.pdf", {"pdf"}),
    ("report_injected.tiff", {"tiff"}),
    ("image_sc_burned.dcm", {"dicom"}),
])
def test_routing(corpus, name, allowed):
    res = route(corpus / name)
    assert res.input_type in allowed, (name, res)
    assert res.error is None


@pytest.mark.parametrize("name", ["notes.txt", "corrupt.png"])
def test_junk_is_unknown_with_error(corpus, name):
    res = route(corpus / name)
    assert res.input_type == "unknown"
    assert res.error
    assert str(corpus) not in res.error  # no absolute paths in error reasons


def test_pdf_loads_two_pages_at_200dpi(corpus):
    assert route(corpus / "report_injected_multipage.pdf").n_pages == 2
    pages = load_pages(corpus / "report_injected_multipage.pdf", "pdf", dpi=200)
    assert [p.index for p in pages] == [1, 2]
    for p in pages:
        w, h = p.original_size
        assert abs(w - PAGE_W) <= 2 and abs(h - PAGE_H) <= 2  # saved at 200 dpi -> same pixel grid
        assert p.meta["dpi"] == 200 and p.transform.is_identity()


def test_tiff_loads_all_frames(corpus):
    pages = load_pages(corpus / "report_injected.tiff", "tiff")
    assert len(pages) == 2
    assert pages[0].original_size == (PAGE_W, PAGE_H)


def test_image_loads_one_page(corpus):
    pages = load_pages(corpus / "report_clean.png", "scan")
    assert len(pages) == 1 and pages[0].original_size == (PAGE_W, PAGE_H)


def test_preamble_less_dicom_is_detected(corpus, tmp_path):
    raw = (corpus / "image_sc_burned.dcm").read_bytes()
    import pydicom

    ds = pydicom.dcmread(corpus / "image_sc_burned.dcm")
    out = tmp_path / "nopreamble"
    ds.save_as(str(out), enforce_file_format=False, implicit_vr=False, little_endian=True)
    data = out.read_bytes()
    if data[128:132] == b"DICM":  # writer kept the preamble: strip it to simulate legacy files
        data = data[132:]
        out.write_bytes(data)
    assert data[128:132] != b"DICM" and raw[128:132] == b"DICM"
    assert route(out).input_type == "dicom"


def test_photo_heuristic_signals():
    rng = np.random.default_rng(140)
    flat = np.full((800, 600), 250, np.uint8)
    assert scan_or_photo(flat)[0] == "scan"
    grad = (np.linspace(0.6, 1.0, 600)[None, :] * 250 * np.ones((800, 1))).astype(np.uint8)
    assert scan_or_photo(grad)[0] == "photo"
    desk = flat.copy()
    desk[:80], desk[-80:], desk[:, :80], desk[:, -80:] = 30, 30, 30, 30
    assert scan_or_photo(desk)[0] == "photo"
    assert scan_or_photo(flat, exif_camera=True)[0] == "photo"
    noisy = np.clip(flat + rng.normal(0, 3, flat.shape), 0, 255).astype(np.uint8)
    assert scan_or_photo(noisy)[0] == "scan"


def test_sniff_magic():
    assert sniff(b"\0" * 128 + b"DICM" + b"\0" * 10) == "dicom"
    assert sniff(b"%PDF-1.7\n") == "pdf"
    assert sniff(b"II*\x00rest") == "tiff" and sniff(b"MM\x00*rest") == "tiff"
    assert sniff(b"\x89PNG\r\n\x1a\nxx") == "image" and sniff(b"\xff\xd8\xff\xe0") == "image"
    assert sniff(b"hello") is None


def test_router_never_raises_on_random_bytes(tmp_path):
    rng = np.random.default_rng(140)
    exts = [".png", ".jpg", ".pdf", ".tiff", ".dcm", ".txt", "", ".bin"]
    prefixes = [b"", b"\x89PNG\r\n\x1a\n", b"%PDF-1.4\n", b"II*\x00", b"\xff\xd8\xff", b"\0" * 128 + b"DICM"]
    for i in range(50):
        n = int(rng.integers(0, 4000))
        data = prefixes[i % len(prefixes)] + rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        p = tmp_path / f"fuzz_{i}{exts[i % len(exts)]}"
        p.write_bytes(data)
        res = route(p)  # must not raise
        assert res.input_type in {"dicom", "pdf", "tiff", "scan", "photo", "unknown"}
        if res.input_type == "unknown":
            assert res.error


def test_missing_and_empty_files_are_unknown(tmp_path):
    assert route(tmp_path / "nope.png").input_type == "unknown"
    (tmp_path / "empty.pdf").write_bytes(b"")
    res = route(tmp_path / "empty.pdf")
    assert res.input_type == "unknown" and res.error
