"""Focused tests of radintake.sdk.Pipeline: typed results, input kinds, write-free default."""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from PIL import Image, ImageDraw

from radintake.config import schema_path
from radintake.pipeline import process_file
from radintake.results import Result
from radintake.sdk import Pipeline, RadintakeConfig

VALIDATOR = Draft202012Validator(json.loads(schema_path("cli_output.v1.json").read_text(encoding="utf-8")))


def _page() -> Image.Image:
    img = Image.new("L", (900, 600), 255)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(["Patient Name: Alva Brennstrom", "FINDINGS:", "Lungs are clear.",
                              "IMPRESSION:", "No acute disease."]):
        d.text((60, 60 + 60 * i), line, fill=0)
    return img


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """cwd, system temp and data dir pointed at empty folders; returns them."""
    boxes = {k: tmp_path / k for k in ("cwd", "tmp", "data")}
    for p in boxes.values():
        p.mkdir()
    monkeypatch.chdir(boxes["cwd"])
    monkeypatch.setenv("TMP", str(boxes["tmp"]))
    monkeypatch.setenv("TEMP", str(boxes["tmp"]))
    monkeypatch.setattr(tempfile, "tempdir", str(boxes["tmp"]))
    monkeypatch.setenv("RADINTAKE_DATA_DIR", str(boxes["data"]))
    monkeypatch.setenv("RADINTAKE_OFFLINE", "1")
    monkeypatch.delenv("RADINTAKE_TRACE_DIR", raising=False)
    return boxes


def _listing(*dirs: Path) -> list[str]:
    return sorted(str(p) for d in dirs for p in d.rglob("*"))


def _view(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k != "trace_id"}


def test_process_path_typed_result_validates(sandbox, tmp_path):
    src = tmp_path / "page.png"
    _page().save(src)
    r = Pipeline(RadintakeConfig(offline=True)).process(src)
    assert isinstance(r, Result)
    assert r.input == "page.png" and r.input_type in ("scan", "photo")
    VALIDATOR.validate(r.model_dump(mode="json"))
    assert len(r.masked_pages) == len(r.phi.masked_artifacts) >= 1
    assert r.masked_pages[0][:8] == b"\x89PNG\r\n\x1a\n"
    assert "masked_pages" not in r.model_dump() and "dicom_bytes" not in r.model_dump()


def test_input_kinds_give_equivalent_results(sandbox, tmp_path):
    img = _page()
    src = tmp_path / "page.png"
    img.save(src)
    pipe = Pipeline(RadintakeConfig(offline=True))
    base = pipe.process(src)
    for source in (src.read_bytes(), img, np.asarray(img), [img]):
        r = pipe.process(source)
        VALIDATOR.validate(r.model_dump(mode="json"))
        assert r.status == base.status
        assert r.flags == base.flags
        assert (r.report.values() if r.report else None) == (base.report.values() if base.report else None)


def test_default_process_is_write_free(sandbox, tmp_path):
    src = tmp_path / "page.png"
    _page().save(src)
    before = _listing(*sandbox.values())
    r = Pipeline(RadintakeConfig(offline=True)).process(_page())
    assert r.status in ("ok", "needs_review")
    assert _listing(*sandbox.values()) == before == []


def test_output_dir_and_trace_dir(sandbox, tmp_path):
    out, traces = tmp_path / "out", tmp_path / "traces"
    pipe = Pipeline(RadintakeConfig(offline=True), trace_dir=traces)
    r = pipe.process(_page(), input_name="scan1.png", output_dir=out)
    doc = json.loads((out / "scan1.png.json").read_text(encoding="utf-8"))
    assert _view(doc) == _view(r.model_dump(mode="json"))
    for rel in r.phi.masked_artifacts:
        assert (out / rel).is_file()
    run_dir = traces / pipe.run_id
    assert (run_dir / "manifest.json").is_file() and list(run_dir.glob("*.jsonl"))
    assert r.trace_id.startswith(pipe.run_id + "/")
    assert _listing(*sandbox.values()) == []


def test_radintake_config_without_yaml(sandbox, tmp_path, monkeypatch):
    empty = tmp_path / "noconf"
    empty.mkdir()
    monkeypatch.setenv("RADINTAKE_CONFIG_DIR", str(empty))
    cfg = RadintakeConfig(offline=True, pdf_dpi=150, budgets={"max_steps": 15})
    pipe = Pipeline(cfg)
    assert pipe.config.pdf_dpi == 150 and pipe.config.budgets.max_steps == 15 and pipe.config.offline
    r = pipe.process(_page())
    VALIDATOR.validate(r.model_dump(mode="json"))
    assert r.status in ("ok", "needs_review")


def test_env_flags_only_tighten(monkeypatch):
    monkeypatch.setenv("RADINTAKE_OFFLINE", "0")
    monkeypatch.setenv("RADINTAKE_FORBID_MODEL_CALLS", "1")
    cfg = RadintakeConfig(offline=True).to_config()
    assert cfg.offline is True and cfg.forbid_model_calls is True


def test_bad_inputs(sandbox):
    pipe = Pipeline(RadintakeConfig(offline=True))
    r = pipe.process(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, input_name="corrupt.png")
    assert r.status == "error" and r.input == "corrupt.png"
    VALIDATOR.validate(r.model_dump(mode="json"))
    with pytest.raises(TypeError):
        pipe.process(object())
    assert _listing(*sandbox.values()) == []


def test_matches_process_file(sandbox, tmp_path):
    src = tmp_path / "page.png"
    _page().save(src)
    cfg = RadintakeConfig(offline=True).to_config()
    legacy = process_file(src, tmp_path / "legacy", config=cfg)
    r = Pipeline(cfg).process(src)
    assert _view(r.model_dump(mode="json")) == _view(legacy)


def test_corpus_parity_and_write_free(sandbox, tmp_path):
    """Every synthetic fixture kind (PNG, photo, PDF, TIFF, DICOM, junk) matches process_file,
    path and bytes give the same result, and nothing is left in cwd / temp / data_dir."""
    from tests.integration.fixtures import build_corpus

    corpus = tmp_path / "corpus"
    build_corpus(corpus)
    cfg = RadintakeConfig(offline=True).to_config()
    legacy_cfg = dataclasses.replace(cfg, data_dir=tmp_path / "legacy_data")
    legacy = {p.name: process_file(p, tmp_path / "legacy", config=legacy_cfg)
              for p in sorted(corpus.iterdir())}
    pipe = Pipeline(cfg)
    before = _listing(*sandbox.values())
    for p in sorted(corpus.iterdir()):
        r = pipe.process(p)
        VALIDATOR.validate(r.model_dump(mode="json"))
        assert _view(r.model_dump(mode="json")) == _view(legacy[p.name]), p.name
        rb = pipe.process(p.read_bytes(), input_name=p.name)
        assert (rb.status, rb.flags, rb.input_type) == (r.status, r.flags, r.input_type), p.name
        if r.input_type == "dicom" and r.status != "error":
            assert rb.dicom_bytes and rb.dicom_bytes[128:132] == b"DICM"
    assert _listing(*sandbox.values()) == before == []
