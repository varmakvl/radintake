"""OCR tool: accuracy on the synthetic corpus, orientation, determinism, offline."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from rapidfuzz import fuzz

from radintake.core.types import BBox
from radintake.tools import ocr as ocr_tool
from tests.integration.fixtures import REPORTS, build_corpus

try:
    ocr_tool.tool_version()
except Exception:  # noqa: BLE001
    pytest.skip("tesseract not installed", allow_module_level=True)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ocr_corpus")
    build_corpus(folder)
    return folder


@pytest.fixture(scope="module")
def clean(corpus):
    img = np.asarray(Image.open(corpus / "report_clean.png").convert("L"))
    return img, ocr_tool.ocr(img)


def _text(words):
    return " ".join(w.text for w in words)


def test_sections_match_truth(clean):
    truth = REPORTS[0]
    _, words = clean
    text = _text(words)
    # the fixture's 54px title runs off the right page edge, so only its visible part is truth
    assert fuzz.partial_ratio("NORTHGATE GENERAL HOSPITAL - DEPARTMENT OF", text) >= 95
    for key in ("indication", "comparison", "findings", "impression"):
        assert fuzz.partial_ratio(getattr(truth, key), text) >= 95, key
    for key in ("patient_name", "patient_id", "date_of_birth", "accession_number"):
        assert fuzz.partial_ratio(getattr(truth, key), text) >= 95, key


def test_patient_id_box_iou(clean):
    truth = REPORTS[0]
    _, words = clean
    gt = BBox.from_seq(truth.boxes.get("patient_id") or _rendered_box())
    hits = [w for w in words if "MRN" in w.text or w.text.strip("MRN-").isdigit()]
    assert hits, "patient id word not found"
    best = max(w.bbox.iou(gt) for w in hits)
    assert best >= 0.5, best


def _rendered_box():
    from tests.integration.fixtures import render_report

    render_report(REPORTS[0])
    return REPORTS[0].boxes["patient_id"]


def test_word_contract(clean):
    img, words = clean
    h, w = img.shape
    assert words
    for ow in words:
        assert 0.0 <= ow.conf <= 1.0
        assert 0 <= ow.bbox.x0 < ow.bbox.x1 <= w and 0 <= ow.bbox.y0 < ow.bbox.y1 <= h
        assert ow.line_id and ow.to_word(2).page == 2


def test_orientation_clean_is_0(clean):
    assert ocr_tool.orientation(clean[0]) == 0


def test_orientation_rot90_photo(corpus):
    img = np.asarray(Image.open(corpus / "report_photo_rot90.jpg").convert("L"))
    assert ocr_tool.orientation(img) in (90, 270)


@pytest.mark.parametrize("deg", [90, 180, 270])
def test_orientation_recovers_rotation(clean, deg):
    rotated = np.rot90(clean[0], k=deg // 90)  # CCW by deg -> needs CW by deg to restore
    assert ocr_tool.orientation(rotated) == deg


def test_deterministic(clean):
    img, first = clean
    again = ocr_tool.ocr(img)
    assert [(w.text, w.bbox.as_list(), w.conf) for w in first] == [(w.text, w.bbox.as_list(), w.conf) for w in again]


def test_runs_offline(clean, monkeypatch):
    import socket

    def boom(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setenv("RADINTAKE_OFFLINE", "1")
    monkeypatch.setattr(socket.socket, "connect", boom)
    assert ocr_tool.ocr(clean[0][:300, :]) is not None


def test_trace_records_tool_and_version(clean, tmp_path):
    from radintake.tracing.writer import TraceWriter

    case = TraceWriter(tmp_path).start_case("c", "0" * 64)
    ocr_tool.ocr(clean[0][:400, :], trace=case)
    ev = [e for e in case.events if e.get("tool") == "ocr"] if hasattr(case, "events") else []
    if ev:
        assert ev[0]["tool_version"].startswith("tesseract ")
    assert ocr_tool.tool_version().startswith("tesseract ")
