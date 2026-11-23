"""Unit tests for the core contracts: config, transforms, trace writer, model client, logging."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from radintake import logging as rlog
from radintake.config import load_config
from radintake.core.types import BBox, Page, Transform
from radintake.tools.model_client import (
    MaskedPayload,
    ModelCallForbidden,
    ModelClient,
    OfflineViolation,
    UnmaskedPayloadError,
)
from radintake.tracing.writer import TraceWriter, hash_text, sanitize

ROOT = Path(__file__).resolve().parents[2]
TRACE_SCHEMA = json.loads((ROOT / "schemas" / "trace.v1.json").read_text())
FAKE_PHI = "Zebulon Fakename-Q7"
FAKE_MRN = "MRN-0000917"
SHA = "a" * 64


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("RADINTAKE_OFFLINE", "RADINTAKE_FORBID_MODEL_CALLS", "RADINTAKE_DATA_DIR",
              "RADINTAKE_TRACE_DIR"):
        monkeypatch.delenv(k, raising=False)
    rlog.clear_phi()
    yield
    rlog.clear_phi()


# ---- config ---------------------------------------------------------------------------
def test_config_reads_yaml_and_env(monkeypatch, tmp_path):
    cfg = load_config()
    assert cfg.offline is False and cfg.forbid_model_calls is False
    assert cfg.pdf_dpi == 200 and cfg.budgets.max_steps > 0
    assert cfg.model("vlm_api").kind == "api" and cfg.model("ocr").kind == "local"

    monkeypatch.setenv("RADINTAKE_OFFLINE", "1")
    monkeypatch.setenv("RADINTAKE_FORBID_MODEL_CALLS", "true")
    monkeypatch.setenv("RADINTAKE_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("RADINTAKE_TRACE_DIR", str(tmp_path / "t"))
    user = tmp_path / "user.yaml"
    user.write_text("budgets:\n  max_steps: 3\n")
    cfg = load_config(user)
    assert cfg.offline and cfg.forbid_model_calls
    assert cfg.data_dir == tmp_path / "d" and cfg.trace_dir == tmp_path / "t"
    assert cfg.budgets.max_steps == 3


# ---- transforms -----------------------------------------------------------------------
def _random_transforms(rng, w, h):
    yield Transform.rot90(int(rng.integers(0, 4)), w, h)
    yield Transform.scale(float(rng.uniform(0.3, 3.0)), float(rng.uniform(0.3, 3.0)))
    yield Transform.translation(float(rng.uniform(-500, 500)), float(rng.uniform(-500, 500)))


def test_transform_compose_inverse_round_trip_boxes_within_1px():
    rng = np.random.default_rng(140)
    w, h = 1700.0, 2200.0
    for _ in range(200):
        x0, y0 = rng.uniform(0, w - 50), rng.uniform(0, h - 50)
        box = BBox(x0, y0, x0 + rng.uniform(5, 400), y0 + rng.uniform(5, 300))
        t = Transform.identity()
        for step in _random_transforms(rng, w, h):
            t = t.compose(step)
        back = t.inverse().map_box(t.map_box(box))
        assert np.allclose(back.as_list(), box.as_list(), atol=1.0)
        # general homography (rotation + perspective): corner points round-trip exactly
        persp = Transform(np.array([[1.0, 0.05, 3], [-0.04, 1.02, 7], [1e-5, 2e-5, 1]]))
        full = t.then(Transform.rotation(float(rng.uniform(-15, 15)), w / 2, h / 2)).then(persp)
        quad = full.map_quad(box)
        assert np.allclose(full.inverse().map_points(quad), box.corners(), atol=1.0)


def test_rot90_matches_numpy_pixels():
    img = np.zeros((40, 60), dtype=np.uint8)
    img[5:10, 20:30] = 255  # box x 20..30, y 5..10
    for k in range(4):
        rot = np.rot90(img, k)
        ys, xs = np.nonzero(rot)
        got = Transform.rot90(k, 60, 40).map_box([20, 5, 30, 10])
        assert np.allclose(got.as_list(), [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])


def test_page_maps_processed_box_back_to_original():
    page = Page(index=1, image=np.zeros((2200, 1700), np.uint8), original_size=(1700, 2200))
    rotated = page.with_image(np.zeros((1700, 2200), np.uint8), Transform.rot90(1, 1700, 2200))
    box = BBox(100, 200, 400, 260)
    assert np.allclose(rotated.to_original(rotated.to_current(box)).as_list(), box.as_list(), atol=1e-6)


# ---- model client ---------------------------------------------------------------------
class _EchoClient(ModelClient):
    def _invoke(self, payload, prompt, **kwargs):
        return {"text": "ok"}


def _masked():
    return MaskedPayload(data={"img": "masked"}, phi_masked=True)


def test_client_raises_on_unmasked_payload():
    c = _EchoClient("ocr")
    with pytest.raises(UnmaskedPayloadError):
        c.call({"data": "x"}, prompt_id="p")
    with pytest.raises(UnmaskedPayloadError):
        c.call(MaskedPayload(data="x"), prompt_id="p")
    assert c.call(_masked(), prompt_id="p") == {"text": "ok"}
    assert c.call({"data": "x", "phi_masked": True}, prompt_id="p") == {"text": "ok"}


def test_client_raises_when_model_calls_forbidden(monkeypatch):
    c = _EchoClient("ocr")
    monkeypatch.setenv("RADINTAKE_FORBID_MODEL_CALLS", "1")
    with pytest.raises(ModelCallForbidden):
        c.call(_masked(), prompt_id="p")


def test_client_refuses_api_models_offline(monkeypatch):
    monkeypatch.setenv("RADINTAKE_OFFLINE", "1")
    with pytest.raises(OfflineViolation):
        _EchoClient("vlm_api").call(_masked(), prompt_id="p")
    assert _EchoClient("ocr").call(_masked(), prompt_id="p") == {"text": "ok"}  # local still fine


def test_client_records_model_call_in_trace(tmp_path):
    tw = TraceWriter(tmp_path, run_id="run1", model_ids={"ocr": "tesseract-5"})
    ct = tw.start_case("case.png", SHA)
    _EchoClient("ocr", trace=ct).call(_masked(), prompt_id="extract_v1", prompt="Extract fields.")
    ct.close("ok")
    lines = [json.loads(x) for x in ct.path.read_text().splitlines()]
    mc = next(x for x in lines if x["kind"] == "model_call")["data"]
    assert mc["model_id"] == "tesseract-5" and mc["prompt_id"] == "extract_v1"
    assert mc["prompt_hash"] == hash_text("Extract fields.") and mc["duration_ms"] >= 0
    man = json.loads(tw.manifest_path.read_text())
    assert man["prompts"]["extract_v1"] == hash_text("Extract fields.")
    assert "tesseract-5" in man["models_called"]


# ---- trace writer ---------------------------------------------------------------------
def test_trace_contains_no_registered_phi_and_validates(tmp_path):
    rlog.register_phi(FAKE_PHI, FAKE_MRN)
    tw = TraceWriter(tmp_path, run_id="run-phi", model_ids={"ocr": "tesseract-5"}, offline=True)
    ct = tw.start_case("report_clean.png", SHA, input_type="scan")
    ct.step("extract", "ok", 12.5, fields={"patient_name": {"value": FAKE_PHI, "score": 0.9}})
    ct.tool_call("ocr", {"text": f"Patient Name: {FAKE_PHI}"}, {"words": [FAKE_PHI, FAKE_MRN]}, 3.0)
    # even a "safe" key gets scrubbed if it carries a registered value
    ct.event("error", error=f"could not parse {FAKE_MRN}", error_type="ValueError")
    ct.event("decision", reason="unregistered free text stays hashed: Jane Unregistered")
    ct.close("needs_review", flags=["low_confidence"])
    tw.finalize()

    blob = "".join(p.read_text() for p in tmp_path.rglob("*") if p.is_file())
    assert FAKE_PHI not in blob and FAKE_MRN not in blob
    assert FAKE_PHI.lower() not in blob.lower()

    validator = Draft202012Validator(TRACE_SCHEMA)
    lines = ct.path.read_text().splitlines()
    assert len(lines) == 6
    for line in lines:
        rec = json.loads(line)
        errors = list(validator.iter_errors(rec))
        assert not errors, [e.message for e in errors]
    # per-field value is a hash, not the value (enough to re-score later)
    step = json.loads(lines[1])
    assert step["data"]["fields"]["patient_name"]["value"] == hash_text(FAKE_PHI)
    assert step["data"]["fields"]["patient_name"]["score"] == 0.9

    man_schema = {**TRACE_SCHEMA["$defs"]["manifest"], "$defs": TRACE_SCHEMA["$defs"]}
    man = json.loads(tw.manifest_path.read_text())
    assert not list(Draft202012Validator(man_schema).iter_errors(man))
    assert man["cases"][0]["status"] == "needs_review"
    assert man["model_ids"] == {"ocr": "tesseract-5"} and man["tool_versions"]["python"]


def test_sanitize_hashes_strings_under_unknown_keys():
    out = sanitize({"stage": "phi", "text": "hello", "n": 3, "items": ["a", 1.5, None]})
    assert out["stage"] == "phi" and out["n"] == 3
    assert out["text"] == hash_text("hello")
    assert out["items"] == [hash_text("a"), 1.5, None]


# ---- logging --------------------------------------------------------------------------
def test_logger_scrubs_registered_phi(capsys):
    rlog.configure_logging("INFO")
    rlog.register_phi(FAKE_PHI)
    log = rlog.get_logger("test")
    log.info("processing", detail=f"name is {FAKE_PHI}", nested={"k": [FAKE_PHI.upper()]})
    log.info(f"event text with {FAKE_PHI}")
    err = capsys.readouterr().err
    assert err.strip(), "nothing logged"
    assert FAKE_PHI.lower() not in err.lower()
    assert rlog.REDACTED in err
    for line in err.strip().splitlines():
        json.loads(line)  # JSON lines


def test_scrubber_hook_is_applied():
    rlog.register_scrubber(lambda s: s.replace("vault-secret", "[VAULT]"))
    assert rlog.scrub("x vault-secret y") == "x [VAULT] y"
