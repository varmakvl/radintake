"""OcrWord contract for both OCR engines and the ocr.engine switch (docs/ocr-engine.md).

The contract test is parametrised over ``tesseract`` and ``paddle``; the paddle case is skipped via
``pytest.importorskip`` when the optional ``[paddle]`` extra is not installed (CI).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from radintake.config import load_config
from radintake.core.types import BBox
from radintake.tools import ocr as ocr_tool
from radintake.tools import ocr_paddle
from tests.integration.fixtures import _font

LINES = ["ACCESSION NUMBER ACC4417", "FINDINGS Lungs are clear bilaterally", "IMPRESSION No acute disease"]


@pytest.fixture(scope="module")
def rendered():
    """White page with text lines; returns (gray array, per-line boxes, ACCESSION box).

    The last line is a form-style label and value with a wide gap, which a line detector such as
    PaddleOCR returns as two boxes - they must still come back as ONE line_id.
    """
    img = Image.new("L", (1400, 530), 255)
    d = ImageDraw.Draw(img)
    font = _font(40)
    lines_boxes = []
    for i, line in enumerate(LINES):
        d.text((60, 60 + i * 110), line, fill=0, font=font)
        lines_boxes.append(d.textbbox((60, 60 + i * 110), line, font=font))
    y = 60 + len(LINES) * 110
    d.text((60, y), "Patient ID:", fill=0, font=font)
    d.text((900, y), "MRN 552013", fill=0, font=font)
    x0, y0, _, y1 = d.textbbox((60, y), "Patient ID:", font=font)
    lines_boxes.append((x0, y0, d.textbbox((900, y), "MRN 552013", font=font)[2], y1))
    acc = d.textbbox((60, 60), "ACCESSION", font=font)
    return np.asarray(img), lines_boxes, BBox(*acc)


@pytest.fixture
def engine(request):
    name = request.param
    if name == "tesseract":
        try:
            ocr_tool.tool_version()
        except Exception:  # noqa: BLE001
            pytest.skip("tesseract not installed")
    else:
        pytest.importorskip("paddleocr")
        pytest.importorskip("paddle")
    ocr_tool.set_engine(name)
    yield name
    ocr_tool.set_engine(None)


@pytest.mark.parametrize("engine", ["tesseract", "paddle"], indirect=True)
def test_ocrword_contract(engine, rendered, tmp_path):
    from radintake.tracing.writer import TraceWriter

    img, line_boxes, acc_box = rendered
    h, w = img.shape
    case = TraceWriter(tmp_path).start_case("c", "0" * 64)
    words = ocr_tool.ocr(img, trace=case)
    assert words, engine
    # the selected engine really ran (no silent fallback to the other one)
    ev = [json.loads(x)["data"] for x in case.path.read_text().splitlines()]
    versions = [e["tool_version"] for e in ev if e.get("tool") == "ocr"]
    assert versions and versions[0].startswith({"tesseract": "tesseract ", "paddle": "paddleocr "}[engine])
    for ow in words:
        assert ow.text.strip()
        assert 0 <= ow.bbox.x0 < ow.bbox.x1 <= w and 0 <= ow.bbox.y0 < ow.bbox.y1 <= h
        assert 0.0 <= ow.conf <= 1.0
        assert all(isinstance(v, int) for v in (ow.block, ow.par, ow.line, ow.word))

    # every rendered line comes back as exactly one line_id whose words read left to right
    by_line: dict[tuple, list] = {}
    for ow in words:
        by_line.setdefault(ow.line_id, []).append(ow)
    for x0, y0, x1, y1 in line_boxes:
        cy = (y0 + y1) / 2
        ids = {ow.line_id for ow in words if ow.bbox.y0 <= cy <= ow.bbox.y1}
        assert len(ids) == 1, (engine, ids)
        row = by_line[ids.pop()]
        xs = [ow.bbox.x0 for ow in row]
        assert xs == sorted(xs), engine
        assert [ow.word for ow in row] == sorted(ow.word for ow in row)
    # reading order: lines top to bottom in the returned sequence
    firsts = [by_line[k][0].bbox.y0 for k in dict.fromkeys(ow.line_id for ow in words)]
    assert firsts == sorted(firsts), engine

    hits = [ow for ow in words if "ACCESSION" in ow.text.upper()]
    assert hits, (engine, [ow.text for ow in words])
    assert max(ow.bbox.iou(acc_box) for ow in hits) > 0.3, engine


@pytest.mark.parametrize("engine", ["paddle"], indirect=True)
def test_paddle_trace_records_engine_version(engine, rendered, tmp_path):
    from radintake.tracing.writer import TraceWriter

    case = TraceWriter(tmp_path).start_case("c", "0" * 64)
    ocr_tool.ocr(rendered[0], trace=case)
    ev = [json.loads(x) for x in case.path.read_text().splitlines()]
    calls = [e["data"] for e in ev if e["kind"] == "tool_call" and e["data"].get("tool") == "ocr"]
    assert calls and calls[0]["tool_version"].startswith("paddleocr ")
    assert calls[0]["output"]["n_words"] > 0


# ---- switch ------------------------------------------------------------------------------
def _config_dir(tmp_path, engine: str):
    from radintake.config import config_dir

    src = config_dir()
    (tmp_path / "models.yaml").write_text((src / "models.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "default.yaml").write_text(f"ocr:\n  engine: {engine}\n", encoding="utf-8")
    return tmp_path


def test_default_is_tesseract_and_manifest_reports_it(monkeypatch, tmp_path):
    from radintake.tracing.writer import TraceWriter

    monkeypatch.delenv("RADINTAKE_OCR_ENGINE", raising=False)
    ocr_tool.set_engine(None)
    cfg = load_config()
    assert cfg.ocr_engine == "tesseract" and ocr_tool.active_engine() == "tesseract"
    assert cfg.model_ids["ocr"] == "tesseract-5" and "ocr_paddle" not in cfg.model_ids
    tw = TraceWriter(tmp_path, run_id="r", model_ids=cfg.model_ids)
    tw.finalize()
    assert json.loads(tw.manifest_path.read_text())["model_ids"]["ocr"] == "tesseract-5"


def test_config_file_selects_engine(monkeypatch, tmp_path):
    monkeypatch.delenv("RADINTAKE_OCR_ENGINE", raising=False)
    monkeypatch.setenv("RADINTAKE_CONFIG_DIR", str(_config_dir(tmp_path, "paddle")))
    ocr_tool.set_engine(None)
    cfg = load_config()
    assert cfg.ocr_engine == "paddle" and ocr_tool.active_engine() == "paddle"
    assert cfg.model_ids["ocr"] == "paddleocr-ppocrv4"


def test_user_yaml_selects_engine(monkeypatch, tmp_path):
    monkeypatch.delenv("RADINTAKE_OCR_ENGINE", raising=False)
    user = tmp_path / "user.yaml"
    user.write_text("ocr:\n  engine: paddle\n", encoding="utf-8")
    assert load_config(user).ocr_engine == "paddle"


def test_env_overrides_config(monkeypatch, tmp_path):
    monkeypatch.setenv("RADINTAKE_CONFIG_DIR", str(_config_dir(tmp_path, "paddle")))
    monkeypatch.setenv("RADINTAKE_OCR_ENGINE", "tesseract")
    ocr_tool.set_engine(None)
    assert load_config().ocr_engine == "tesseract" and ocr_tool.active_engine() == "tesseract"
    tess_dir = tmp_path / "t"
    tess_dir.mkdir()
    monkeypatch.setenv("RADINTAKE_CONFIG_DIR", str(_config_dir(tess_dir, "tesseract")))
    monkeypatch.setenv("RADINTAKE_OCR_ENGINE", "paddle")
    assert ocr_tool.active_engine() == "paddle"
    assert load_config().model_ids["ocr"] == "paddleocr-ppocrv4"


def test_set_engine_pins_over_env(monkeypatch):
    monkeypatch.setenv("RADINTAKE_OCR_ENGINE", "paddle")
    try:
        ocr_tool.set_engine("tesseract")
        assert ocr_tool.active_engine() == "tesseract"
    finally:
        ocr_tool.set_engine(None)
    assert ocr_tool.active_engine() == "paddle"


def test_unknown_engine_rejected(monkeypatch):
    monkeypatch.setenv("RADINTAKE_OCR_ENGINE", "easyocr")
    with pytest.raises(ValueError, match="ocr.engine"):
        load_config()
    with pytest.raises(ValueError):
        ocr_tool.set_engine("easyocr")


def test_missing_paddle_extra_raises_clear_error(monkeypatch, rendered):
    """Documented behaviour: no silent Tesseract fallback, OcrUnavailable naming the extra."""
    monkeypatch.setattr(ocr_paddle, "available", lambda: False)
    monkeypatch.setattr(ocr_paddle, "_engine", None)
    ocr_tool.set_engine("paddle")
    try:
        with pytest.raises(ocr_tool.OcrUnavailable, match=r"\[paddle\].*requirements-paddle\.lock"):
            ocr_tool.ocr(rendered[0])
    finally:
        ocr_tool.set_engine(None)


def test_pipeline_pins_engine_from_config(monkeypatch, tmp_path):
    """process_file pins cfg.ocr_engine, so a --config choice reaches every ocr() call of the run."""
    from radintake import pipeline

    seen = []
    monkeypatch.setattr(ocr_tool, "set_engine", lambda name: seen.append(name))
    user = tmp_path / "u.yaml"
    user.write_text("ocr:\n  engine: paddle\n", encoding="utf-8")
    missing = tmp_path / "nope.png"
    pipeline.process_file(missing, tmp_path / "out", config=load_config(user), write=False)
    assert seen == ["paddle"]


# ---- paddle post-processing (pure; runs without the extra) ---------------------------------
def test_split_words_proportional_boxes():
    words = ocr_paddle.split_words(BBox(100, 10, 300, 40), "AB  CDEF")
    assert [t for t, _ in words] == ["AB", "CDEF"]
    (_, a), (_, b) = words
    assert a.x0 == 100 and a.x1 == 150 and b.x0 == 200 and b.x1 == 300
    assert a.y0 == b.y0 == 10 and a.y1 == b.y1 == 40


def test_rows_merge_same_line_boxes_and_order():
    raw = [[
        [[[400, 12], [600, 12], [600, 42], [400, 42]], ("Smith", 0.9)],
        [[[10, 100], [200, 100], [200, 130], [10, 130]], ("Second line", 0.8)],
        [[[10, 10], [300, 10], [300, 40], [10, 40]], ("Patient Name:", 1.2)],
        [[[-5, 200], [2000, 200], [2000, 230], [-5, 230]], ("wide", 0.5)],
    ]]
    lines = ocr_paddle._lines(raw, 1000, 220)
    assert lines[2][2] == 1.0  # conf clamped to 0..1
    assert lines[3][0] == BBox(0, 200, 1000, 220)  # clamped to the image
    rows = ocr_paddle._rows(lines)
    assert [[t for _, t, _ in r] for r in rows] == [["Patient Name:", "Smith"], ["Second line"], ["wide"]]
    assert ocr_paddle._lines([None], 10, 10) == []


def test_bgr_conversion():
    gray = np.zeros((2, 3), np.uint8)
    assert ocr_paddle._bgr(gray).shape == (2, 3, 3)
    rgb = np.zeros((1, 1, 3), np.uint8)
    rgb[0, 0] = (1, 2, 3)
    assert tuple(ocr_paddle._bgr(rgb)[0, 0]) == (3, 2, 1)
