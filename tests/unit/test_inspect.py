"""Step inspector: per-stage dumps when enabled, a no-op otherwise, never able to fail a case."""

from __future__ import annotations

import shutil

import pytest

from radintake.orchestrator import planner


def test_inspect_hook_is_a_noop_without_env(monkeypatch):
    monkeypatch.delenv("RADINTAKE_INSPECT_DIR", raising=False)
    called = []
    monkeypatch.setattr("radintake.inspect.dump_stage", lambda *a, **k: called.append(a))
    planner._inspect("load", object(), {}, 0.0)
    assert called == []


def test_inspect_failure_never_propagates(monkeypatch, tmp_path):
    monkeypatch.setenv("RADINTAKE_INSPECT_DIR", str(tmp_path))

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr("radintake.inspect.dump_stage", boom)
    planner._inspect("load", object(), {}, 0.0)  # must not raise


@pytest.mark.skipif(shutil.which("tesseract") is None and not __import__("os").environ.get("TESSERACT_CMD"),
                    reason="tesseract not installed")
def test_inspector_dumps_every_document_step(tmp_path, monkeypatch):
    from radintake import inspect
    from tests.integration.fixtures import REPORTS, render_report

    src = tmp_path / "in"
    src.mkdir()
    render_report(REPORTS[0]).save(src / "page.png")
    monkeypatch.setenv("RADINTAKE_OFFLINE", "1")
    monkeypatch.setenv("RADINTAKE_DATA_DIR", str(tmp_path / "data"))
    assert inspect.main([str(src), "--out", str(tmp_path / "out")]) == 0
    case = tmp_path / "out" / "page.png"
    for step, files in {"00_input": ["page_1.png"], "20_preprocess": ["page_1.png", "decisions.json"],
                        "30_phi": ["page_1_masked.png", "page_1_ocr.png", "detections.json", "ocr_words_1.json"],
                        "40_extract": ["report.json", "page_1_fields.png"], "50_verify": ["fields.json"]}.items():
        for f in files:
            assert (case / step / f).is_file(), f"{step}/{f} missing"
    assert "40_extract" in (case / "index.html").read_text(encoding="utf-8")
