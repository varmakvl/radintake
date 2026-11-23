"""Document PHI agent: layered detection, image/text masking, vault, scrubber, guarded client."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

import radintake.logging as rlog
from radintake.agents import phi as phi_mod
from radintake.agents.phi import PhiDetector, PhiVault, mask_image, mask_page, mask_text
from radintake.config import load_config
from radintake.core.types import BBox, Page, Transform, Word
from radintake.orchestrator.stages import DOCUMENT_TYPES, StageContext, discover, registered
from radintake.tools import ocr as ocr_tool
from radintake.tools.model_client import MaskedPayload, ModelClient, UnmaskedPayloadError
from radintake.tracing.writer import TraceWriter
from tests.integration.fixtures import PAGE_H, PAGE_W, REPORTS, degrade_photo, render_report

try:
    ocr_tool.tool_version()
    _HAVE_OCR = True
except Exception:  # noqa: BLE001
    _HAVE_OCR = False
needs_ocr = pytest.mark.skipif(not _HAVE_OCR, reason="tesseract not installed")

HEADER = ("patient_name", "patient_id", "date_of_birth", "accession_number")
TYPE_OF = {"patient_name": "PATIENT_NAME", "patient_id": "PATIENT_ID",
           "date_of_birth": "DATE_OF_BIRTH", "accession_number": "ACCESSION_NUMBER"}


@pytest.fixture(autouse=True)
def _clean_registry():
    rlog.clear_phi()
    yield
    rlog.clear_phi()


def _synthetic_page() -> tuple[Page, list[Word]]:
    """A tiny white page with dark 'word' blocks and hand-made OCR words (no Tesseract needed)."""
    img = np.full((300, 900), 255, np.uint8)
    rows = [
        [("Patient", 20), ("Name:", 140), ("Marlowe", 300), ("Quistgaard", 450)],
        [("Patient", 20), ("ID:", 140), ("MRN-7730215", 300)],
        [("DOB:", 20), ("1961-04-17", 300)],
        [("Accession", 20), ("No.:", 180), ("ACC-55019384", 300)],
        [("Seen", 20), ("by", 110), ("Dr.", 170), ("Halvorsen", 240), ("on", 420), ("2024-06-11.", 480)],
    ]
    words: list[Word] = []
    for r, row in enumerate(rows):
        y = 20 + r * 55
        for text, x in row:
            w = 12 * len(text)
            img[y + 4:y + 26, x + 2:x + w - 2] = 0
            words.append(Word(text=text, bbox=BBox(x, y, x + w, y + 30), confidence=0.95))
    return Page(index=1, image=img, original_size=(900, 300)), words


# --------------------------------------------------------------------------- text layer
def test_free_text_doctor_name_and_date_masked_and_restored():
    vault = PhiVault()
    text = "Findings discussed with Dr. Evander Holloway on 2024-06-11 by telephone."
    masked, spans = mask_text(text, vault, PhiDetector())
    assert "Evander" not in masked and "Holloway" not in masked and "2024-06-11" not in masked
    assert "[PERSON_1]" in masked and "[DATE_1]" in masked
    assert masked.startswith("Findings discussed with Dr. [PERSON_1] on [DATE_1]")
    assert {s.type for s in spans} == {"PERSON", "DATE"}
    assert vault.restore(masked) == text
    # values reach the log/trace scrubber; the vault never prints them
    assert rlog.scrub("called Evander Holloway") == f"called {rlog.REDACTED}"
    assert "Holloway" not in repr(vault)


def test_header_anchors_typed_and_patient_name_masked_everywhere():
    vault = PhiVault()
    text = ("Patient Name: Marlowe Quistgaard\nPatient ID: MRN-7730215\nDate of Birth: 1961-04-17\n"
            "Accession No.: ACC-55019384\nQuistgaard tolerated the exam. Report by Jane Moller, MD.")
    masked, _ = mask_text(text, vault)
    for v in ("Marlowe", "Quistgaard", "MRN-7730215", "1961-04-17", "ACC-55019384", "Jane Moller"):
        assert v not in masked, v
    for ph in ("[PATIENT_NAME_1]", "[PATIENT_ID_1]", "[DATE_OF_BIRTH_1]", "[ACCESSION_NUMBER_1]", "[PERSON_1]"):
        assert ph in masked, ph
    assert vault.value("[PATIENT_ID_1]") == "MRN-7730215"
    assert vault.values_of("ACCESSION_NUMBER") == ["ACC-55019384"]
    assert vault.restore(masked) == text


def test_same_value_reuses_placeholder():
    v = PhiVault()
    assert v.placeholder("DATE", "2024-06-11") == v.placeholder("date", " 2024-06-11 ") == "[DATE_1]"
    assert v.placeholder("DATE", "2019-03-02") == "[DATE_2]"


# --------------------------------------------------------------------------- page layer
def test_mask_page_on_synthetic_words_blanks_boxes_and_keeps_original():
    page, words = _synthetic_page()
    before = page.image.copy()
    res = mask_page(page, None, words, PhiDetector(), PhiVault(), pad=4)
    assert np.array_equal(page.image, before), "source page must never be mutated"
    types = {d.type for d in res.detections}
    assert {"PATIENT_NAME", "PATIENT_ID", "DATE_OF_BIRTH", "ACCESSION_NUMBER", "PERSON", "DATE"} <= types
    for w in words:
        region = res.masked_original[int(w.bbox.y0):int(w.bbox.y1), int(w.bbox.x0):int(w.bbox.x1)]
        if w.text in ("Marlowe", "Quistgaard", "MRN-7730215", "1961-04-17", "ACC-55019384",
                      "Halvorsen", "2024-06-11."):
            assert region.std() < 10, w.text
        elif w.text in ("Patient", "Accession", "Seen"):
            assert region.std() > 50, f"label {w.text} must stay visible"
    t = res.masked_text
    assert "Dr. [PERSON_1]" in t and "[DATE_1]." in t
    assert res.payload.phi_masked is True and res.payload.data["text"] == t


def test_mask_image_pads_and_copies():
    img = np.zeros((50, 50), np.uint8)
    out = mask_image(img, [BBox(10, 10, 20, 20)], pad=3, fill=200)
    assert img.max() == 0
    assert (out[7:23, 7:23] == 200).all() and out[6, 6] == 0 and out[23, 23] == 0


def test_detections_mapped_to_original_coords():
    page, words = _synthetic_page()
    # pretend the processed page is the original rotated 90 deg clockwise (rot90 k=3)
    rot = Transform.rot90(3, 900, 300)
    proc = page.with_image(np.ascontiguousarray(np.rot90(page.image, 3)), rot)
    pwords = [Word(w.text, rot.map_box(w.bbox), w.confidence) for w in words]
    res = mask_page(proc, page, pwords, PhiDetector(), PhiVault())
    by_type = {d.type: d for d in res.detections}
    pid = by_type["PATIENT_ID"].bbox
    assert abs(pid.x0 - 300) < 1 and abs(pid.y0 - 75) < 1 and abs(pid.x1 - 432) < 1
    out = by_type["PATIENT_ID"].to_output()
    assert out["page"] == 1 and len(out["bbox"]) == 4


# --------------------------------------------------------------------------- guarded model client
class _MockVlm(ModelClient):
    def __init__(self, **kw):
        super().__init__("mock-vlm", kind="local", **kw)
        self.seen: list = []

    def _invoke(self, payload, prompt, **kwargs):
        self.seen.append(payload)
        return {"ok": True}


def test_model_client_refuses_unmasked_page_and_accepts_masked(monkeypatch):
    monkeypatch.delenv("RADINTAKE_FORBID_MODEL_CALLS", raising=False)
    page, words = _synthetic_page()
    client = _MockVlm(config=load_config())
    with pytest.raises(UnmaskedPayloadError):
        client.call({"image": page.image, "text": "Patient Name: Marlowe Quistgaard"}, prompt_id="p")
    with pytest.raises(UnmaskedPayloadError):
        client.call(MaskedPayload(data={"image": page.image}), prompt_id="p")
    assert client.seen == []
    res = mask_page(page, None, words, PhiDetector(), PhiVault())
    assert client.call(res.payload, prompt_id="p") == {"ok": True}
    assert client.seen == [res.payload]
    assert "Quistgaard" not in client.seen[0].data["text"]


# --------------------------------------------------------------------------- stage
def test_phi_stage_registered_before_extract():
    discover()
    st = registered()["phi"]
    assert st.applies_to == DOCUMENT_TYPES
    assert registered()["preprocess"].order < st.order
    assert st.failure_flag == "phi_uncertain"
    for name, other in registered().items():
        if "extract" in name:
            assert other.order > st.order


def test_phi_stage_writes_artifacts_and_schema_shaped_output(tmp_path, monkeypatch):
    page, words = _synthetic_page()
    monkeypatch.setattr(ocr_tool, "ocr", lambda img, **kw: list(words))
    tw = TraceWriter(tmp_path / "_traces")
    trace = tw.start_case("doc.png", "0" * 64)
    ctx = StageContext(path=tmp_path / "doc.png", input_name="doc.png", input_type="scan",
                       input_sha256="0" * 64, out_dir=tmp_path, config=load_config(), trace=trace,
                       deadline=1e18, data={"pages": [page]})
    out = phi_mod.phi_stage(ctx)
    trace.close("ok")
    assert out["phi"]["masked_artifacts"] == ["doc.png.masked/page_1.png"]
    art = np.asarray(Image.open(tmp_path / "doc.png.masked" / "page_1.png"))
    assert art.shape == page.image.shape
    for d in out["phi"]["detections"]:
        assert set(d) == {"type", "page", "bbox"} and d["page"] == 1
    assert all(p.phi_masked is True for p in out["phi_masked_pages"])
    assert out["pages"][0].words and out["flags"] == []
    blob = "".join(p.read_text() for p in (tmp_path / "_traces").rglob("*.jsonl"))
    blob += json.dumps(out["phi"])
    for v in ("Marlowe", "Quistgaard", "MRN-7730215", "1961-04-17", "ACC-55019384", "Halvorsen"):
        assert v not in blob, v


# --------------------------------------------------------------------------- recall on fixtures
def _coverage(gt, box) -> float:
    ix = max(0.0, min(gt[2], box[2]) - max(gt[0], box[0]))
    iy = max(0.0, min(gt[3], box[3]) - max(gt[1], box[1]))
    return ix * iy / ((gt[2] - gt[0]) * (gt[3] - gt[1]))


@needs_ocr
@pytest.mark.parametrize("name", ["report_clean.png", "report_photo_rot90.jpg"])
def test_header_phi_recall_on_fixtures(name, tmp_path):
    from radintake.agents.preprocess import preprocess_page

    t = REPORTS[0]
    img = render_report(t)
    gt = dict(t.boxes)
    if name.endswith(".jpg"):
        path = tmp_path / name
        degrade_photo(img).save(path, quality=60)
        img = Image.open(path)
        rot = Transform.rot90(1, PAGE_W, PAGE_H)  # PIL rotate(90, expand=True)
        gt = {k: tuple(rot.map_box(v).as_list()) for k, v in gt.items()}
    arr = np.asarray(img.convert("L"))
    original = Page(index=1, image=arr, original_size=(arr.shape[1], arr.shape[0]))
    proc = preprocess_page(original).page
    res = mask_page(proc, original, ocr_tool.ocr(proc.image), PhiDetector(), PhiVault())
    hits = 0
    for key in HEADER:
        boxes = [d.bbox.as_list() for d in res.detections if d.type == TYPE_OF[key]]
        if boxes and max(_coverage(gt[key], b) for b in boxes) >= 0.7:
            hits += 1
    assert hits == 4, f"{name}: header PHI recall {hits}/4"
    if name == "report_clean.png":
        for key in HEADER:
            x0, y0, x1, y1 = (int(v) for v in gt[key])
            assert res.masked_original[y0:y1, x0:x1].std() < 10, key
    for v in t.phi_strings:
        assert v not in res.masked_text

