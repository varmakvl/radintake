"""End-to-end integration test: batch CLI on an unseen, mixed input folder.

Mirrors the reviewers' hidden test: the CLI is pointed at a folder it has never seen
(PNG / JPEG photo / multi-page PDF / multi-page TIFF / DICOM / junk) and must write one
schema-valid JSON per input. Runs the REAL pipeline as a subprocess (no mocks).

External model APIs: if RADINTAKE_OFFLINE=1 the pipeline must run with local models only.
CI sets it so this test is deterministic and free.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from rapidfuzz import fuzz

from .fixtures import build_corpus

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "schemas" / "cli_output.v1.json").read_text())
TEXT_FIELDS = ("indication", "comparison", "findings", "impression")


@pytest.fixture(scope="module")
def batch_run(tmp_path_factory):
    inp = tmp_path_factory.mktemp("unseen_inputs")
    out = tmp_path_factory.mktemp("outputs")
    truth = build_corpus(inp)
    env = {**os.environ, "RADINTAKE_OFFLINE": os.environ.get("RADINTAKE_OFFLINE", "1"),
           "RADINTAKE_TRACE_DIR": str(out / "_traces"),
           "RADINTAKE_DATA_DIR": str(out / "_data")}
    proc = subprocess.run(
        [sys.executable, "-m", "radintake.cli", "batch", "--in", str(inp), "--out", str(out)],
        capture_output=True, text=True, env=env, timeout=1800, cwd=ROOT, check=False,
    )
    return {"inp": inp, "out": out, "truth": truth, "proc": proc}


def _load(batch_run, name: str) -> dict:
    p = batch_run["out"] / f"{name}.json"
    assert p.exists(), f"missing output JSON for {name}"
    return json.loads(p.read_text())


def test_cli_exits_cleanly(batch_run):
    proc = batch_run["proc"]
    assert proc.returncode == 0, f"CLI failed:\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}"


def test_one_schema_valid_json_per_input(batch_run):
    inputs = sorted(p.name for p in batch_run["inp"].iterdir() if p.is_file())
    validator = Draft202012Validator(SCHEMA)
    for name in inputs:
        doc = _load(batch_run, name)
        errors = sorted(validator.iter_errors(doc), key=lambda e: e.path)
        assert not errors, f"{name}: schema errors: {[e.message for e in errors]}"
        expected_sha = hashlib.sha256((batch_run["inp"] / name).read_bytes()).hexdigest()
        assert doc["input_sha256"] == expected_sha
        assert doc["input"] == name


def test_input_routing(batch_run):
    expect = {
        "report_clean.png": {"scan"},
        "report_photo_rot90.jpg": {"photo", "scan"},
        "report_injected_multipage.pdf": {"pdf"},
        "report_injected.tiff": {"tiff"},
        "image_sc_burned.dcm": {"dicom"},
    }
    for name, allowed in expect.items():
        assert _load(batch_run, name)["input_type"] in allowed, name


def test_junk_inputs_produce_error_json_not_crash(batch_run):
    for name in batch_run["truth"]["junk"]:
        doc = _load(batch_run, name)
        assert doc["status"] == "error"
        assert doc.get("error")


@pytest.mark.parametrize("name", ["report_clean.png", "report_photo_rot90.jpg",
                                  "report_injected_multipage.pdf", "report_injected.tiff"])
def test_report_extraction_fields_and_provenance(batch_run, name):
    truth = batch_run["truth"]["reports"][name]
    doc = _load(batch_run, name)
    assert doc["status"] in ("ok", "needs_review")
    rep = doc["report"]
    assert rep is not None
    min_ratio = 95 if name == "report_clean.png" else 85
    for f in TEXT_FIELDS:
        got = rep[f]["value"] or ""
        assert fuzz.ratio(got.lower(), getattr(truth, f).lower()) >= min_ratio, (name, f, got)
        assert 0.0 <= rep[f]["confidence"] <= 1.0
        prov = rep[f]["provenance"]
        assert prov, f"{name}:{f} has no provenance"
        for p in prov:
            x0, y0, x1, y1 = p["bbox"]
            assert x0 < x1 and y0 < y1 and p["page"] >= 1
    # header IDs extracted locally (exact match on clean page)
    if name == "report_clean.png":
        assert rep["patient_id"]["value"] == truth.patient_id
        assert rep["accession_number"]["value"] == truth.accession_number


def test_provenance_maps_to_original_page_coordinates(batch_run):
    """On the clean page, cited boxes must overlap the ground-truth boxes (IoU >= 0.3)."""
    truth = batch_run["truth"]["reports"]["report_clean.png"]
    rep = _load(batch_run, "report_clean.png")["report"]

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua else 0.0

    for f in TEXT_FIELDS + ("patient_id", "accession_number"):
        boxes = [p["bbox"] for p in rep[f]["provenance"] if p["page"] == 1]
        assert max(iou(b, truth.boxes[f]) for b in boxes) >= 0.3, f


def test_photo_provenance_maps_to_original_page_coordinates(batch_run):
    """The rotated phone photo is read upright after preprocessing; its cited boxes must still be
    in the photo's OWN pixel grid (IoU >= 0.3 with the truth boxes rotated like the photo)."""
    from radintake.core.types import BBox, Transform

    from .fixtures import PAGE_H, PAGE_W

    truth = batch_run["truth"]["reports"]["report_photo_rot90.jpg"]
    rep = _load(batch_run, "report_photo_rot90.jpg")["report"]
    rot = Transform.rot90(1, PAGE_W, PAGE_H)  # degrade_photo: PIL rotate(90, expand=True)
    for f in TEXT_FIELDS + ("patient_id", "accession_number"):
        want = rot.map_box(truth.boxes[f])
        boxes = [BBox.from_seq(p["bbox"]) for p in rep[f]["provenance"] if p["page"] == 1]
        assert boxes and max(b.iou(want) for b in boxes) >= 0.3, f


@pytest.mark.parametrize("name", ["report_injected_multipage.pdf", "report_injected.tiff"])
def test_prompt_injection_is_treated_as_data(batch_run, name):
    truth = batch_run["truth"]["reports"][name]
    doc = _load(batch_run, name)
    rep = doc["report"]
    # impression must be the real one, not "normal"
    assert fuzz.ratio(rep["impression"]["value"].lower(), truth.impression.lower()) >= 85
    assert "normal" not in rep["impression"]["value"].lower().replace("within normal limits", "")
    assert "prompt_injection_suspected" in doc.get("flags", [])
    assert doc["n_pages"] == 2


@pytest.mark.parametrize("name", ["report_injected_multipage.pdf", "report_injected.tiff"])
def test_injection_review_queue_entry_without_phi(batch_run, name):
    from radintake.store import review_queue

    items = [i for i in review_queue.list_items(batch_run["out"] / "_data") if i["case_id"] == name]
    assert items and "prompt_injection_suspected" in items[0]["flags"]
    jl = (batch_run["out"] / "review_queue.jsonl").read_text()
    assert name in jl
    blob = json.dumps(items) + jl
    for t in batch_run["truth"]["reports"].values():
        for s in t.phi_strings:
            assert s not in blob, f"PHI in review queue: {s!r}"


def test_no_phi_in_logs_or_stdout(batch_run):
    proc = batch_run["proc"]
    blob = proc.stdout + proc.stderr
    trace_dir = batch_run["out"] / "_traces"
    for p in trace_dir.rglob("*") if trace_dir.exists() else []:
        if p.is_file():  # every file under _traces, whatever its suffix
            blob += p.read_bytes().decode("utf-8", errors="ignore")
    for t in batch_run["truth"]["reports"].values():
        for s in t.phi_strings:
            assert s not in blob, f"PHI leaked into logs/traces: {s!r}"


def test_traces_written_and_replayable(batch_run):
    trace_dir = batch_run["out"] / "_traces"
    assert trace_dir.exists() and any(trace_dir.rglob("*.jsonl")), "no traces written"
    proc = subprocess.run(
        [sys.executable, "-m", "radintake.eval.rescore", "--traces", str(trace_dir), "--offline"],
        capture_output=True, text=True, cwd=ROOT, timeout=300, check=False,
        env={**os.environ, "RADINTAKE_FORBID_MODEL_CALLS": "1"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_phi_masked_page_artifacts(batch_run):
    """Masked page images exist and header PHI regions are blanked (near-uniform)."""
    from PIL import Image

    truth = batch_run["truth"]["reports"]["report_clean.png"]
    doc = _load(batch_run, "report_clean.png")
    arts = doc["phi"]["masked_artifacts"]
    assert arts
    masked = np.asarray(Image.open(batch_run["out"] / arts[0]).convert("L"), dtype=np.float32)
    for key in ("patient_name", "patient_id", "date_of_birth", "accession_number"):
        x0, y0, x1, y1 = truth.boxes[key]
        region = masked[y0:y1, x0:x1]
        assert region.std() < 10, f"{key} not masked"


def test_dicom_deidentified_and_valid(batch_run):
    import pydicom

    dt_ = batch_run["truth"]["dicom"]["image_sc_burned.dcm"]
    doc = _load(batch_run, "image_sc_burned.dcm")
    assert doc["dicom"]["valid"] is True
    out_path = batch_run["out"] / doc["dicom"]["output_path"]
    ds = pydicom.dcmread(str(out_path))  # must still parse as DICOM
    for tag, val in dt_.phi_values.items():
        assert str(getattr(ds, tag, "") or "") != val, f"{tag} not de-identified"
    assert getattr(ds, "PatientIdentityRemoved", "") == "YES"

    src = pydicom.dcmread(str(dt_.path)).pixel_array.astype(np.int16)
    red = ds.pixel_array.astype(np.int16)
    assert red.shape == src.shape
    x0, y0, x1, y1 = dt_.burned_box
    # burned-in text removed
    assert np.abs(red[y0:y1, x0:x1] - src[y0:y1, x0:x1]).mean() > 20
    # rest of image untouched (allow a margin around the redaction box)
    mask = np.ones_like(src, dtype=bool)
    mask[max(0, y0 - 15):y1 + 15, max(0, x0 - 15):x1 + 15] = False
    assert np.array_equal(red[mask], src[mask]), "pixels outside burned-in region were altered"
