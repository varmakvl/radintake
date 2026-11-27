"""DICOM de-identification: header profile, consistent UID remap, burned-in redaction, re-encode."""

from __future__ import annotations

import json

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, RLELossless, SecondaryCaptureImageStorage, generate_uid

import radintake.logging as rlog
from radintake.agents import dicom_deid as dd
from radintake.config import load_config
from radintake.core.types import BBox
from radintake.orchestrator.stages import DICOM_TYPES, StageContext, discover, registered
from radintake.tools import dicom_io
from radintake.tools import ocr as ocr_tool
from radintake.tools.ocr import OcrWord
from radintake.tracing.writer import TraceWriter
from tests.integration.fixtures import make_dicom

try:
    ocr_tool.tool_version()
    _HAVE_OCR = True
except Exception:  # noqa: BLE001
    _HAVE_OCR = False
needs_ocr = pytest.mark.skipif(not _HAVE_OCR, reason="tesseract not installed")


@pytest.fixture(autouse=True)
def _clean_registry():
    rlog.clear_phi()
    yield
    rlog.clear_phi()


def _outside_equal(src: np.ndarray, red: np.ndarray, regions) -> bool:
    keep = np.ones(src.shape[:3], dtype=bool)
    for i, b in regions:
        keep[i, int(b.y0):int(np.ceil(b.y1)), int(b.x0):int(np.ceil(b.x1))] = False
    return np.array_equal(src[keep], red[keep])


def _ds(path, *, study_uid: str, series_uid: str | None = None, frames: int = 1,
        burned: bool = False) -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), Dataset(), file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.PatientName, ds.PatientID, ds.Modality = "Doe^Jane", "PID-0042", "OT"
    ds.BurnedInAnnotation = "YES" if burned else "NO"
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 120, (frames, 64, 96), dtype=np.uint8)
    if burned:
        arr[-1, 10:20, 10:60] = 255
    ds.Rows, ds.Columns = 64, 96
    if frames > 1:
        ds.NumberOfFrames = frames
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit, ds.PixelRepresentation = 7, 0
    ds.PixelData = (arr if frames > 1 else arr[0]).tobytes()
    return ds


def _stub_ocr(img, **kw):
    """Pretend OCR: reads one word where the binarised mask has a solid ink run >= 40 px.

    Scattered bright noise (what real OCR rejects) produces no word.
    """
    ink = (np.asarray(img) == 0).astype(np.int32)
    run = np.apply_along_axis(lambda r: np.convolve(r, np.ones(40, np.int32), "valid"), 1, ink) == 40
    ys, xs = np.nonzero(run)
    if not len(xs):
        return []
    xs = np.concatenate([xs, xs + 39])
    return [OcrWord("TEXT", BBox(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), 0.9, 1, 1, 1, 1)]


# --------------------------------------------------------------------------- header / UIDs
def test_uids_remapped_consistently_across_two_instances_of_one_study(tmp_path):
    study, series = generate_uid(), generate_uid()
    remap = dd.UidRemapper(salt=b"test")
    outs = []
    for k in range(2):
        src = tmp_path / f"i{k}.dcm"
        _ds(src, study_uid=study, series_uid=series).save_as(str(src), enforce_file_format=True)
        res = dd.deidentify_file(src, tmp_path / f"o{k}.dcm", remap=remap, ocr_fn=_stub_ocr)
        assert res.valid
        outs.append(pydicom.dcmread(str(res.output)))
    a, b = outs
    assert a.StudyInstanceUID == b.StudyInstanceUID != study
    assert a.SeriesInstanceUID == b.SeriesInstanceUID != series
    assert a.SOPInstanceUID != b.SOPInstanceUID
    assert a.SOPClassUID == SecondaryCaptureImageStorage  # registered UIDs are kept
    for ds in outs:
        assert str(ds.file_meta.MediaStorageSOPInstanceUID) == str(ds.SOPInstanceUID)
        assert str(ds.StudyInstanceUID).startswith("2.25.") and len(str(ds.StudyInstanceUID)) <= 64
    # a different remapper (another process) does not reproduce the mapping
    assert dd.UidRemapper()(study) != remap(study)


def test_private_tags_dropped_and_unlisted_person_names_emptied(tmp_path):
    ds = _ds(tmp_path / "p.dcm", study_uid=generate_uid())
    block = ds.private_block(0x0011, "ACME 1.0", create=True)
    block.add_new(0x01, "LO", "Jane Doe secret")
    item = Dataset()
    item.PersonName = "Smith^Alan"  # PN inside a sequence, not in the profile table
    ds.ContentSequence = [item]
    ds.InstitutionName = "Northgate General Hospital"
    tags = dd.deidentify_header(ds, dd.load_profile(), dd.UidRemapper(b"x"))
    assert not any(e.tag.is_private for e in ds)
    assert str(ds.ContentSequence[0].PersonName) == ""
    assert "InstitutionName" not in ds and ds.PatientName == "" and ds.PatientID == ""
    assert {"PatientName", "PatientID", "InstitutionName", "PersonName"} <= tags
    # every removed/replaced value is now scrubbed from logs and traces
    assert rlog.scrub("x Jane Doe secret y") == f"x {rlog.REDACTED} y"
    assert rlog.scrub("Northgate General Hospital") == rlog.REDACTED


def test_private_safe_list_keeps_listed_element(tmp_path):
    ds = _ds(tmp_path / "p.dcm", study_uid=generate_uid())
    block = ds.private_block(0x0011, "ACME 1.0", create=True)
    block.add_new(0x01, "LO", "kept")
    block.add_new(0x02, "LO", "dropped")
    prof = {**dd.load_profile(), "private_safe_list": [{"creator": "ACME 1.0", "element": "01"}]}
    dd.deidentify_header(ds, prof, dd.UidRemapper(b"x"))
    assert [str(e.value) for e in ds if e.tag.is_private] == ["ACME 1.0", "kept"]


def test_pad_capped_at_15():
    assert dd.load_profile(type("C", (), {"raw": {"dicom_deid": {"burned_in": {"pad_px": 40}}}})())[
        "burned_in"]["pad_px"] == 15


# --------------------------------------------------------------------------- pixels
def test_multiframe_only_text_frame_redacted_outside_bit_identical(tmp_path):
    src = tmp_path / "mf.dcm"
    _ds(src, study_uid=generate_uid(), frames=3, burned=True).save_as(str(src), enforce_file_format=True)
    orig = dicom_io.frames(pydicom.dcmread(str(src)))
    res = dd.deidentify_file(src, tmp_path / "mf.out.dcm", ocr_fn=_stub_ocr)
    red = dicom_io.frames(pydicom.dcmread(str(res.output)))
    assert res.valid and red.shape == orig.shape == (3, 64, 96)
    assert {i for i, _ in res.regions} == {2}
    assert np.array_equal(red[:2], orig[:2])
    assert _outside_equal(orig, red, res.regions)
    assert (red[2, 10:20, 10:60] == red[2].min()).all()
    assert pydicom.dcmread(str(res.output)).BurnedInAnnotation == "NO"


def test_declared_burned_in_but_nothing_found_is_uncertain(tmp_path):
    src = tmp_path / "u.dcm"
    _ds(src, study_uid=generate_uid(), burned=True).save_as(str(src), enforce_file_format=True)
    res = dd.deidentify_file(src, tmp_path / "u.out.dcm", ocr_fn=lambda img, **kw: [])
    assert res.uncertain and not res.regions
    assert pydicom.dcmread(str(res.output)).BurnedInAnnotation == "YES"  # never claim NO falsely


@needs_ocr
def test_fixture_header_phi_removed_and_burned_text_redacted(tmp_path):
    t = make_dicom(tmp_path / "f.dcm")
    src_ds = pydicom.dcmread(str(t.path))
    res = dd.deidentify_file(t.path, tmp_path / "f.deid.dcm")
    out = pydicom.dcmread(str(res.output))
    for tag, val in t.phi_values.items():
        assert str(getattr(out, tag, "") or "") != val, tag
    assert out.PatientIdentityRemoved == "YES" and out.BurnedInAnnotation == "NO"
    assert out.DeidentificationMethod
    assert [c.CodeValue for c in out.DeidentificationMethodCodeSequence] == ["113100", "113101"]
    assert out.StudyInstanceUID != src_ds.StudyInstanceUID
    src, red = dicom_io.frames(src_ds), dicom_io.frames(out)
    assert _outside_equal(src, red, res.regions)
    x0, y0, x1, y1 = t.burned_box
    assert np.abs(red[0, y0:y1, x0:x1].astype(int) - src[0, y0:y1, x0:x1].astype(int)).mean() > 20
    for _, b in res.regions:  # stays inside the 15 px margin the integration test allows
        assert b.x0 >= x0 - 15 and b.y0 >= y0 - 15 and b.x1 <= x1 + 15 and b.y1 <= y1 + 15


@needs_ocr
def test_rle_compressed_dicom_handled(tmp_path):
    t = make_dicom(tmp_path / "plain.dcm")
    ds = pydicom.dcmread(str(t.path))
    ds.compress(RLELossless, generate_instance_uid=False)
    rle = tmp_path / "rle.dcm"
    ds.save_as(str(rle), enforce_file_format=True)
    assert str(pydicom.dcmread(str(rle)).file_meta.TransferSyntaxUID) == RLELossless
    res = dd.deidentify_file(rle, tmp_path / "rle.deid.dcm")
    out = pydicom.dcmread(str(res.output))
    assert res.valid and res.reencoded
    assert str(out.file_meta.TransferSyntaxUID) == RLELossless  # original syntax kept
    src, red = dicom_io.frames(pydicom.dcmread(str(rle))), dicom_io.frames(out)
    assert red.shape == src.shape and res.regions
    assert _outside_equal(src, red, res.regions)
    assert out.PatientIdentityRemoved == "YES" and out.BurnedInAnnotation == "NO"


# --------------------------------------------------------------------------- stage
def test_stage_registered_for_dicom():
    discover()
    st = registered()["dicom_deid"]
    assert st.applies_to == DICOM_TYPES and st.failure_flag == "phi_uncertain"


def test_stage_output_and_traces_carry_no_phi(tmp_path, monkeypatch):
    src = tmp_path / "in" / "s.dcm"
    src.parent.mkdir()
    ds = _ds(src, study_uid=generate_uid(), burned=True)
    ds.AccessionNumber, ds.InstitutionName = "ACC-99887766", "Northgate General Hospital"
    ds.save_as(str(src), enforce_file_format=True)
    tw = TraceWriter(tmp_path / "_traces")
    trace = tw.start_case("s.dcm", "0" * 64)
    ctx = StageContext(path=src, input_name="s.dcm", input_type="dicom", input_sha256="0" * 64,
                       out_dir=tmp_path / "out", config=load_config(), trace=trace, deadline=1e18)
    monkeypatch.setattr(dd, "_default_ocr", _stub_ocr)
    out = dd.dicom_deid_stage(ctx)
    trace.close("ok")
    assert out["dicom"]["output_path"] == "s.dcm.deid.dcm" and out["dicom"]["valid"] is True
    assert out["phi"]["masked_artifacts"] == ["s.dcm.deid.dcm"]
    assert (tmp_path / "out" / "s.dcm.deid.dcm").is_file()
    assert out["dicom"]["burned_in_regions"] and out["flags"] == []
    blob = json.dumps(out) + "".join(p.read_text() for p in (tmp_path / "_traces").rglob("*.jsonl"))
    for v in ("Doe^Jane", "PID-0042", "ACC-99887766", "Northgate General Hospital"):
        assert v not in blob, v


# --------------------------------------------------------------------------- text candidates (rad-s2g.5.1)
def _glyph_frame(text: str, org=(40, 60), scale=1.2, size=(120, 320)) -> np.ndarray:
    import cv2

    img = np.full(size, 20, np.uint8)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 2, cv2.LINE_8)
    return img


def test_keep_list_lives_in_profile():
    safe = {s.upper() for s in dd.load_profile()["burned_in"]["safe_text"]}
    assert {"PORTABLE", "SEMI-UPRIGHT", "L", "R"} <= safe


def test_short_token_found_by_candidates_without_ocr_word_boxes():
    """Full-frame OCR sees nothing; the top-hat candidate plus a 1-glyph crop read is enough."""
    img = _glyph_frame("M")

    def ocr_fn(im, *, psm, **kw):
        return [] if psm == 11 else [OcrWord("M", BBox(0, 0, 10, 10), 0.9, 1, 1, 1, 1)]

    boxes = dd.find_text_boxes(img, dd.load_profile()["burned_in"], ocr_fn=ocr_fn)
    ys, xs = np.nonzero(img == 255)
    assert len(boxes) == 1
    b = boxes[0]
    assert b.x0 <= xs.min() and b.y0 <= ys.min() and b.x1 >= xs.max() + 1 and b.y1 >= ys.max() + 1


def test_safe_marker_line_not_redacted():
    img = _glyph_frame("L")

    def ocr_fn(im, *, psm, **kw):
        return [] if psm == 11 else [OcrWord("L", BBox(0, 0, 10, 10), 0.95, 1, 1, 1, 1)]

    assert dd.find_text_boxes(img, dd.load_profile()["burned_in"], ocr_fn=ocr_fn) == []


# --------------------------------------------------------------------------- burned-in v2 on a synthetic DICOM (rad-s2g.5.3)
_NAME_BOX = (20, 14, 330, 50)    # x0, y0, x1, y1 of the synthetic patient-name line
_TOKEN_BOX = (492, 118, 552, 156)  # "[M]" alone on its row (right border)
_MARKER_BOX = (20, 214, 200, 252)  # keep-list marker "PORTABLE"


def _synthetic_burned_dicom(path) -> tuple[FileDataset, np.ndarray]:
    """12-bit MONOCHROME2 'radiograph': textured anatomy in the middle, dark borders carrying burned-in
    text drawn with cv2.putText at full intensity (no TCIA data)."""
    import cv2

    rng = np.random.default_rng(11)
    h, w = 272, 560
    img = np.full((h, w), 60, np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    body = ((xx - w / 2) / 200.0) ** 2 + ((yy - h / 2) / 90.0) ** 2 <= 1.0
    img[body] = 1400 + 900 * np.sin(xx[body] / 9.0) * np.cos(yy[body] / 13.0)
    img += rng.normal(0, 25, img.shape)
    arr = np.clip(img, 0, 3000).astype(np.uint16)
    ink = np.zeros((h, w), np.uint8)  # cv2.putText draws on 8-bit images only
    for text, (x0, y0, x1, y1) in (("DOE JOHN 01.02.1970", _NAME_BOX), ("[M]", _TOKEN_BOX),
                                   ("PORTABLE", _MARKER_BOX)):
        cv2.putText(ink, text, (x0 + 4, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_8)
    arr[ink > 0] = 4095
    ds = _ds(path, study_uid=generate_uid(), burned=True)
    ds.Rows, ds.Columns = h, w
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
    ds.PatientName, ds.PatientID, ds.AccessionNumber = "DOE^JOHN", "PID-77001", "ACC-5512"
    ds.PixelData = arr.tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return ds, arr


def _ink(arr: np.ndarray, box) -> int:
    x0, y0, x1, y1 = box
    return int((arr[y0:y1, x0:x1] >= 4000).sum())


@needs_ocr
def test_v2_name_and_short_token_removed_marker_kept_outside_identical(tmp_path):
    src = tmp_path / "burned.dcm"
    _, arr = _synthetic_burned_dicom(src)
    assert _ink(arr, _NAME_BOX) > 300 and _ink(arr, _TOKEN_BOX) > 50 and _ink(arr, _MARKER_BOX) > 300
    res = dd.deidentify_file(src, tmp_path / "burned.deid.dcm", remap=dd.UidRemapper(b"t"))
    out = pydicom.dcmread(str(res.output))
    red = dicom_io.frames(out)[0]
    assert _ink(red, _NAME_BOX) == 0, "synthetic patient name left in the pixels"
    assert _ink(red, _TOKEN_BOX) == 0, "short token [M] left in the pixels"
    x0, y0, x1, y1 = _MARKER_BOX
    assert np.array_equal(red[y0:y1, x0:x1], arr[y0:y1, x0:x1]), "keep-list marker PORTABLE was altered"
    assert _outside_equal(arr[None], red[None], res.regions)


@needs_ocr
def test_v2_output_valid_dicom_same_pixels_layout_and_header_profile(tmp_path):
    src = tmp_path / "burned.dcm"
    _synthetic_burned_dicom(src)
    src_ds = pydicom.dcmread(str(src))
    res = dd.deidentify_file(src, tmp_path / "burned.deid.dcm", remap=dd.UidRemapper(b"t"))
    out = pydicom.dcmread(str(res.output))
    assert res.valid and res.regions and not res.uncertain
    assert out.pixel_array.shape == src_ds.pixel_array.shape and out.pixel_array.dtype == src_ds.pixel_array.dtype
    assert str(out.file_meta.TransferSyntaxUID) == str(src_ds.file_meta.TransferSyntaxUID)
    assert out.PhotometricInterpretation == src_ds.PhotometricInterpretation
    assert len(out.PixelData) == len(src_ds.PixelData)
    assert out.BurnedInAnnotation == "NO" and out.PatientIdentityRemoved == "YES"
    assert [c.CodeValue for c in out.DeidentificationMethodCodeSequence] == ["113100", "113101"]
    # header PHI handled per configs/dicom_profile.yaml (Z = zero-length)
    actions = dd.load_profile()["actions"]
    for kw in ("PatientName", "PatientID", "AccessionNumber"):
        assert actions[kw] == "Z" and str(getattr(out, kw)) == ""
    assert out.StudyInstanceUID != src_ds.StudyInstanceUID


def test_metrics_aggregate_lists_per_file_dicom_rows():
    from radintake.eval.metrics import aggregate

    base = {"kind": "dicom", "valid": True, "phi_tags": 4, "uids": 2, "uids_kept": 0, "burned_gt": 1}
    recs = [{**base, "dicom_id": "DICOM_A", "modality": "CR", "phi_tags_removed": 4, "leaked_tags": [],
             "burned_hit": 1, "burned_text_px": 100, "burned_text_px_removed": 90,
             "burned_px_changed": 120, "burned_px_changed_in_gt": 96},
            {**base, "dicom_id": "DICOM_B", "modality": "MG", "phi_tags_removed": 3, "leaked_tags": ["StudyDate"],
             "burned_hit": 0, "burned_px_changed": 0, "burned_px_changed_in_gt": 0}]  # legacy: no text mask
    rows = {r["dicom_id"]: r for r in aggregate(recs)["dicom"]["per_file"]}
    assert set(rows) == {"DICOM_A", "DICOM_B"}
    a, b = rows["DICOM_A"], rows["DICOM_B"]
    assert (a["text_pixel_recall"], a["text_pixel_precision"], a["residual_text_px"]) == (0.9, 0.8, 10)
    assert a["changed_px_outside_gt"] == 24 and a["modality"] == "CR" and a["leaked_tags"] == []
    assert b["text_pixel_recall"] is None and b["text_pixel_precision"] is None and b["residual_text_px"] == 0
    assert b["leaked_tags"] == ["StudyDate"]
    for r in rows.values():  # PHI-free: ids, numbers and tag keywords only
        assert set(r) == {"dicom_id", "modality", "valid", "burned_gt_boxes", "burned_hit", "text_pixel_recall",
                          "text_pixel_precision", "residual_text_px", "changed_px", "changed_px_outside_gt",
                          "leaked_tags"}
