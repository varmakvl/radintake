"""Every public entry point in ``radintake.__all__``: typed results, write-free default, guard.

Synthetic inputs only (rendered text pages and the integration fixture corpus); offline.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from PIL import Image, ImageDraw

import radintake
from radintake.config import schema_path
from radintake.tools.model_client import (
    MaskedPayload,
    ModelClient,
    OfflineViolation,
    UnmaskedPayloadError,
)

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = Draft202012Validator(json.loads(schema_path("cli_output.v1.json").read_text(encoding="utf-8")))
PUBLIC = {"__version__", "Pipeline", "process", "aprocess", "RadintakeConfig", "BudgetConfig",
          "ModelConfig", "Result", "Report", "ReportField", "Provenance", "Phi", "Detection",
          "Dicom", "Table", "TableCell", "RadintakeError", "UnsupportedInputError", "ConfigError"}


def _page(text: str = "No acute disease.") -> Image.Image:
    img = Image.new("L", (900, 600), 255)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(["Patient Name: Alva Brennstrom", "FINDINGS:", "Lungs are clear.",
                              "IMPRESSION:", text]):
        d.text((60, 60 + 60 * i), line, fill=0)
    return img


def _validate(r: radintake.Result) -> None:
    VALIDATOR.validate(r.model_dump(mode="json"))


def _key(r: radintake.Result) -> tuple:
    return r.status, tuple(r.flags), r.input_type, r.report.values() if r.report else None


def _listing(*dirs: Path) -> list[str]:
    return sorted(str(p) for d in dirs for p in d.rglob("*"))


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """cwd, TMP/TEMP, RADINTAKE_DATA_DIR and RADINTAKE_TRACE_DIR pointed at empty folders."""
    boxes = {k: tmp_path / k for k in ("cwd", "tmp", "data", "env_traces")}
    for p in boxes.values():
        p.mkdir()
    monkeypatch.chdir(boxes["cwd"])
    monkeypatch.setenv("TMP", str(boxes["tmp"]))
    monkeypatch.setenv("TEMP", str(boxes["tmp"]))
    monkeypatch.setenv("TMPDIR", str(boxes["tmp"]))
    monkeypatch.setattr(tempfile, "tempdir", str(boxes["tmp"]))
    monkeypatch.setenv("RADINTAKE_DATA_DIR", str(boxes["data"]))
    monkeypatch.setenv("RADINTAKE_TRACE_DIR", str(boxes["env_traces"]))
    monkeypatch.setenv("RADINTAKE_OFFLINE", "1")
    return boxes


@pytest.fixture
def pipe():
    return radintake.Pipeline(radintake.RadintakeConfig(offline=True))


# ----------------------------------------------------------------------------- surface
def test_all_is_exactly_the_documented_surface():
    assert set(radintake.__all__) == PUBLIC
    doc = (ROOT / "docs" / "api-stability.md").read_text(encoding="utf-8")
    for name in radintake.__all__:
        assert getattr(radintake, name) is not None
        assert f"`{name}`" in doc, f"{name} missing from docs/api-stability.md"
    with pytest.raises(AttributeError):
        radintake.not_a_public_name  # noqa: B018


def test_import_radintake_is_light():
    code = ("import radintake, sys; [getattr(radintake, n) for n in radintake.__all__]; "
            "print(any(m in sys.modules for m in ('fastapi', 'uvicorn', 'fastembed')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                         check=True)
    assert out.stdout.strip() == "False"


# ----------------------------------------------------------------------------- process
def test_pipeline_process_input_kinds(sandbox, tmp_path, pipe):
    img = _page()
    src = tmp_path / "page.png"
    img.save(src)
    base = pipe.process(src)
    assert isinstance(base, radintake.Result) and base.input == "page.png"
    _validate(base)
    assert isinstance(base.report, radintake.Report)
    assert isinstance(base.report.impression, radintake.ReportField)
    assert isinstance(base.phi, radintake.Phi) and base.masked_pages
    for source in (src.read_bytes(), img, np.asarray(img), [img]):
        r = pipe.process(source)
        _validate(r)
        assert _key(r)[:2] == _key(base)[:2] and _key(r)[3] == _key(base)[3]
    multi = pipe.process([img, _page("Second page.")], input_name="two.tiff")
    _validate(multi)
    assert multi.input == "two.tiff" and multi.input_type == "tiff" and multi.n_pages == 2
    assert _listing(*sandbox.values()) == []


def test_process_convenience_with_output_and_trace_dir(sandbox, tmp_path):
    out, traces = tmp_path / "out", tmp_path / "traces"
    r = radintake.process(_page(), config=radintake.RadintakeConfig(offline=True),
                          input_name="scan.png", output_dir=out, trace_dir=traces)
    _validate(r)
    written = json.loads((out / "scan.png.json").read_text(encoding="utf-8"))
    assert written == r.model_dump(mode="json")
    for rel in r.phi.masked_artifacts:
        assert (out / rel).is_file()
    assert list(traces.glob("*/manifest.json")) and list(traces.glob("*/*.jsonl"))
    assert _listing(*sandbox.values()) == []  # nothing outside output_dir / trace_dir


def test_default_process_is_write_free(sandbox, tmp_path, pipe):
    """Image (masked PNG), DICOM (de-identified DICOM) and junk: nothing left anywhere."""
    from tests.integration.fixtures import make_dicom

    dcm = tmp_path / "in" / "burned.dcm"
    dcm.parent.mkdir()
    make_dicom(dcm)
    before = _listing(*sandbox.values())
    r_img = pipe.process(_page())
    r_dcm = pipe.process(dcm.read_bytes(), input_name="burned.dcm")
    r_bad = pipe.process(b"not a supported input")
    assert r_img.masked_pages and r_img.masked_pages[0][:8] == b"\x89PNG\r\n\x1a\n"
    assert r_dcm.input_type == "dicom" and r_dcm.dicom_bytes[128:132] == b"DICM"
    assert r_bad.status == "error"
    for r in (r_img, r_dcm, r_bad):
        _validate(r)
        assert "masked_pages" not in r.model_dump() and "dicom_bytes" not in r.model_dump()
    assert _listing(*sandbox.values()) == before == []


# ----------------------------------------------------------------------------- batch / async
def test_process_batch_one_result_per_input(sandbox, tmp_path, pipe):
    out = tmp_path / "out"
    seen: list[radintake.Result] = []
    corrupt = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    results = pipe.process_batch([_page(), corrupt, object(), tmp_path / "missing.png"],
                                 output_dir=out, on_result=seen.append)
    assert len(results) == 4 and seen == results
    assert results[0].status in ("ok", "needs_review")
    assert [r.status for r in results[1:]] == ["error", "error", "error"]
    assert results[2].error.startswith("unsupported source")
    for r in results:
        _validate(r)
        assert (out / f"{r.input}.json").is_file()
    assert _listing(*sandbox.values()) == []


def test_async_entry_points(sandbox, pipe):
    async def main():
        one = await pipe.aprocess(_page(), input_name="a.png")
        many = await pipe.aprocess_batch([_page(), b"junk"])
        conv = await radintake.aprocess(b"junk", config=radintake.RadintakeConfig(offline=True))
        return one, many, conv

    one, many, conv = asyncio.run(main())
    assert one.input == "a.png" and one.status in ("ok", "needs_review")
    assert [r.status for r in many][1] == "error" and len(many) == 2
    assert conv.status == "error"
    for r in (one, conv, *many):
        _validate(r)
    assert _listing(*sandbox.values()) == []


# ----------------------------------------------------------------------------- config
def test_radintake_config_without_yaml(sandbox, tmp_path, monkeypatch):
    empty = tmp_path / "noconf"
    empty.mkdir()
    monkeypatch.setenv("RADINTAKE_CONFIG_DIR", str(empty))
    cfg = radintake.RadintakeConfig(
        offline=True, pdf_dpi=150, seed=7,
        budgets=radintake.BudgetConfig(max_steps=12, stage_timeout_s=60.0),
        models={"vlm": radintake.ModelConfig(id="qwen2.5vl:7b", kind="local")})
    p = radintake.Pipeline(cfg)
    assert (p.config.pdf_dpi, p.config.seed, p.config.budgets.max_steps) == (150, 7, 12)
    assert p.config.model("vlm").id == "qwen2.5vl:7b" and p.config.offline
    r = p.process(_page())
    _validate(r)
    assert r.status in ("ok", "needs_review")
    with pytest.raises(Exception):  # noqa: B017 - frozen model
        cfg.offline = False


def test_public_exceptions(pipe):
    with pytest.raises(radintake.UnsupportedInputError) as ei:
        pipe.process(object())
    assert isinstance(ei.value, TypeError) and isinstance(ei.value, radintake.RadintakeError)
    with pytest.raises(radintake.ConfigError):
        radintake.Pipeline(config=42)
    assert issubclass(radintake.ConfigError, radintake.RadintakeError)


def test_result_models_build_a_schema_valid_document():
    prov = radintake.Provenance(page=1, bbox=[0.1, 0.1, 0.5, 0.2], source="ocr")
    field = radintake.ReportField(value="No acute disease.", confidence=0.9, provenance=[prov])
    empty = radintake.ReportField(value=None, confidence=0.0)
    report = radintake.Report(**{k: empty for k in ("patient_name", "patient_id", "date_of_birth",
                                                    "accession_number", "indication", "comparison",
                                                    "findings")}, impression=field)
    cell = radintake.TableCell(row=0, col=0, text="Na", bbox=[0.1, 0.1, 0.2, 0.2], confidence=0.8)
    table = radintake.Table(page=1, bbox=[0.1, 0.1, 0.9, 0.5], n_rows=1, n_cols=1, method="ruled",
                            text_source="ocr", confidence=0.8, cells=[cell])
    r = radintake.Result(input="x.png", input_sha256="a" * 64, input_type="scan", status="ok",
                         trace_id="run/x.png", n_pages=1, report=report,
                         phi=radintake.Phi(masked_artifacts=["x.png/page_1.png"],
                                           detections=[radintake.Detection(type="name", page=1)]),
                         dicom=radintake.Dicom(output_path="x.dcm", valid=True), tables=[table])
    doc = r.model_dump(mode="json")
    VALIDATOR.validate(doc)
    assert doc["error"] is None and doc["report"]["patient_name"]["value"] is None
    assert radintake.Result.model_validate(doc).model_dump(mode="json") == doc


# ----------------------------------------------------------------------------- guard
class _EchoClient(ModelClient):
    def _invoke(self, payload, prompt, **kwargs):
        return {"text": "ok"}


def test_model_guard_still_applies_to_sdk_config(monkeypatch):
    monkeypatch.delenv("RADINTAKE_OFFLINE", raising=False)
    cfg = radintake.Pipeline(radintake.RadintakeConfig(
        offline=True, models={"vlm": radintake.ModelConfig(id="remote-vlm", kind="api"),
                              "ocr": radintake.ModelConfig(id="tesseract", kind="local")})).config
    with pytest.raises(UnmaskedPayloadError):
        _EchoClient("ocr", config=cfg).call({"data": "raw page"}, prompt_id="p")
    with pytest.raises(OfflineViolation):
        _EchoClient("vlm", config=cfg).call(MaskedPayload(data="x", phi_masked=True), prompt_id="p")
    ok = _EchoClient("ocr", config=cfg).call(MaskedPayload(data="x", phi_masked=True), prompt_id="p")
    assert ok == {"text": "ok"}
