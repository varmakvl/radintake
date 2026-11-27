"""PHI layer 4 (handwritten sign-offs): crop selection, masking order, masking and clean fallback.

The handwriting model is replaced by stubs, so CI needs neither transformers nor weights. Every
OCR word of the synthetic page is painted with its OWN grey level, so a stub can tell exactly which
words a crop shows; rule-masked words are filled with the page background (white) before cropping.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import radintake.logging as rlog
from radintake.agents import phi as phi_mod
from radintake.agents.phi import PhiDetector, PhiVault, mask_page
from radintake.config import load_config
from radintake.core.types import BBox, Page, Transform, Word
from radintake.orchestrator.stages import StageContext
from radintake.tools import handwriting as hw_mod
from radintake.tools import ocr as ocr_tool
from radintake.tools.handwriting import HandwritingReader, NonLocalPhiReader, PhiCrop
from radintake.tools.model_client import MaskedPayload, UnmaskedPayloadError
from radintake.tracing.writer import TraceWriter

NAME = "L. Brooks, RT"  # what the stub "recognises" in the sign-off crop (synthetic)
W, H = 1000, 520

# (text, x, confidence) per line; y = 20 + 55 * row. Gaps >= 24 px so crop padding never reaches
# the neighbour word.
ROWS = [
    [("Patient", 20, .95), ("Name:", 140, .95), ("Marlowe", 260, .95), ("Quistgaard", 380, .95)],
    [("FINDINGS:", 20, .95), ("Lungs", 160, .95), ("clear;", 250, .95), ("Kxq", 350, .20)],   # low conf, NOT sign-off zone
    [("IMPRESSION:", 20, .95), ("No", 180, .95), ("acute", 230, .95), ("disease.", 320, .95)],
    [("Technologist", 20, .95), ("note:", 190, .95), ("stand.", 270, .95), ("=", 370, .70),
     ("Le", 410, .05), ("Broo", 470, .90), ("ks,", 550, .90), ("RT", 610, .95)],            # credential anchor
    [("Seen", 20, .95), ("by", 100, .95), ("Dr.", 150, .95), ("Halvorsen", 210, .95)],         # rule-masked
    [("Copy", 20, .95), ("Zwq", 110, .10), ("Vrr", 180, .15), ("to", 260, .95), ("file", 320, .95)],  # low conf in zone
    [("Page", 20, .95), ("1", 110, .95)],
]


def _grey(k: int) -> int:
    return 3 + 7 * k  # 3 .. ~200: distinct per word, never the white background


def _page() -> tuple[Page, list[Word]]:
    img = np.full((H, W), 255, np.uint8)
    words: list[Word] = []
    for r, row in enumerate(ROWS):
        y = 20 + 55 * r
        for text, x, conf in row:
            w = 14 * len(text) + 6
            img[y + 4:y + 26, x + 2:x + w - 2] = _grey(len(words))
            words.append(Word(text=text, bbox=BBox(x, y, x + w, y + 30), confidence=conf))
    return Page(index=1, image=img, original_size=(W, H)), words


def _ids(*texts: str) -> set[int]:
    _, words = _page()
    return {i for i, w in enumerate(words) if w.text in texts}


def _words_in(img: np.ndarray, n: int) -> set[int]:
    return {k for k in range(n) if (np.asarray(img) == _grey(k)).any()}


class StubReader:
    """Stands in for TrOCR. Records what it saw; returns NAME for the credential crop."""

    model_id = "stub-hw"

    def __init__(self, n_words: int, rule_ids: set[int], *, text: str = NAME):
        self.n_words, self.rule_ids, self.text = n_words, rule_ids, text
        self.seen: list[set[int]] = []
        self.rules_ran_first: list[bool] = []

    def read(self, crop, *, trace=None):
        assert isinstance(crop, PhiCrop) and crop.phi_crop is True
        shown = _words_in(crop.image, self.n_words)
        self.seen.append(shown)
        # the masking rules already ran: no rule-detected word is visible in any crop
        self.rules_ran_first.append(_STATE["rules_done"] and not (shown & self.rule_ids))
        return self.text if shown & _ids("Broo") else "illegible"


_STATE = {"rules_done": False}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    rlog.clear_phi()
    _STATE["rules_done"] = False
    real = phi_mod.detect_words

    def spy(*a, **k):  # marks the moment the rule layers (1-3) finished on the page
        out = real(*a, **k)
        _STATE["rules_done"] = True
        return out

    monkeypatch.setattr(phi_mod, "detect_words", spy)
    yield
    rlog.clear_phi()


def _rule_ids(words) -> set[int]:
    hits = phi_mod.detect_words(words, PhiDetector())
    _STATE["rules_done"] = False
    return {i for h in hits for s in h.spans for i in h.words_of(s)}


def _pass(reader) -> phi_mod.HandwritingPass:
    return phi_mod.HandwritingPass(reader, phi_mod.handwriting_params(phi_mod.load_params()))


# --------------------------------------------------------------------------- crop selection
def test_crops_come_only_from_anchor_neighbourhoods_and_low_conf_signoff_lines():
    page, words = _page()
    rule_ids = _rule_ids(words)
    assert _ids("Marlowe", "Quistgaard", "Halvorsen") <= rule_ids  # layers 1-3 did their part
    stub = StubReader(len(words), rule_ids)
    mask_page(page, None, words, PhiDetector(), PhiVault(), handwriting=_pass(stub))
    # exactly two crops: the name left of the credential, and the low-confidence sign-off-zone run
    assert stub.seen == [_ids("Le", "Broo", "ks,", "RT"), _ids("Copy", "Zwq", "Vrr", "to")]
    everything = set().union(*stub.seen)
    assert not everything & _ids("Kxq")          # low confidence but ABOVE the sign-off zone
    assert not everything & _ids("stand.", "=")  # left of the separator: not part of the name
    assert not everything & rule_ids             # already masked by a rule


def test_masking_rules_run_before_any_crop_is_read():
    page, words = _page()
    stub = StubReader(len(words), _rule_ids(words))
    mask_page(page, None, words, PhiDetector(), PhiVault(), handwriting=_pass(stub))
    assert stub.rules_ran_first and all(stub.rules_ran_first)


# --------------------------------------------------------------------------- masking
def test_recognised_name_is_masked_in_original_coordinates():
    page, words = _page()
    rule_ids = _rule_ids(words)
    # the processed page is the original upscaled 2x (e.g. small-text upsampling) and shifted;
    # words and crops are in processed coords, detections must come back in ORIGINAL coords
    step = Transform.scale(2.0).then(Transform.translation(15, 9))
    big = np.full((2 * H + 9, 2 * W + 15), 255, np.uint8)
    big[9:, 15:] = np.repeat(np.repeat(page.image, 2, axis=0), 2, axis=1)
    proc = page.with_image(big, step)
    pwords = [Word(w.text, step.map_box(w.bbox), w.confidence) for w in words]
    stub = StubReader(len(words), rule_ids)
    res = mask_page(proc, page, pwords, PhiDetector(), PhiVault(), handwriting=_pass(stub))
    hw = [d for d in res.detections if d.source == "handwriting"]
    assert len(hw) == 1 and hw[0].type == "PERSON"
    # the sign-off "name, RT" is one detection (the credential is masked with the name)
    signoff = [words[i].bbox for i in sorted(_ids("Le", "Broo", "ks,", "RT"))]
    want = BBox(min(b.x0 for b in signoff), min(b.y0 for b in signoff),
                max(b.x1 for b in signoff), max(b.y1 for b in signoff))
    got = hw[0].bbox
    assert max(abs(a - b) for a, b in zip(got.as_list(), want.as_list())) < 1.0, (got, want)
    # the sign-off pixels are gone from the masked ORIGINAL page; the note before the dash is kept
    for i in _ids("Le", "Broo", "ks,", "RT"):
        b = words[i].bbox
        assert (res.masked_original[int(b.y0):int(b.y1), int(b.x0):int(b.x1)] == 255).all(), words[i].text
    for i in _ids("stand.", "note:"):
        b = words[i].bbox
        assert (res.masked_original[int(b.y0):int(b.y1), int(b.x0):int(b.x1)] == _grey(i)).any(), words[i].text
    assert "Broo" not in res.masked_text and "[PERSON_" in res.masked_text and "stand." in res.masked_text


# --------------------------------------------------------------------------- fallback
def _run_stage(tmp_path, monkeypatch, words, page, sub: str, phi_cfg: dict | None = None):
    monkeypatch.setattr(ocr_tool, "ocr", lambda img, **kw: list(words))
    out_dir = tmp_path / sub
    tw = TraceWriter(out_dir / "_traces")
    trace = tw.start_case("doc.png", "0" * 64)
    cfg = load_config()
    if phi_cfg is not None:
        cfg.raw["phi"] = phi_cfg
    ctx = StageContext(path=out_dir / "doc.png", input_name="doc.png", input_type="scan", input_sha256="0" * 64,
                       out_dir=out_dir, config=cfg, trace=trace, deadline=1e18, data={"pages": [page]})
    out = phi_mod.phi_stage(ctx)
    trace.close("ok")
    events = [json.loads(line) for p in (out_dir / "_traces").rglob("*.jsonl") for line in p.read_text().splitlines()]
    return out, events, out_dir


@pytest.mark.parametrize("missing", ["transformers", "torch", "weights"])
def test_missing_transformers_or_weights_records_skip_and_output_is_unchanged(tmp_path, monkeypatch, missing):
    page, words = _page()
    off, _, off_dir = _run_stage(tmp_path, monkeypatch, words, page, "off", {"handwriting": {"enabled": False}})

    import importlib.util

    real_spec = importlib.util.find_spec

    def find_spec(name, *a, **k):
        if name == missing:
            return None  # == ImportError at load time
        if name in ("transformers", "torch"):
            return object()  # pretend installed
        return real_spec(name, *a, **k)

    monkeypatch.setattr(hw_mod.importlib.util, "find_spec", find_spec)
    if missing == "weights":
        huggingface_hub = pytest.importorskip("huggingface_hub")
        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    reason = {"transformers": "transformers_missing", "torch": "torch_missing", "weights": "weights_not_cached"}[missing]
    assert hw_mod.availability("microsoft/trocr-small-handwritten") == reason

    on, events, on_dir = _run_stage(tmp_path, monkeypatch, words, page, "on")
    skips = [e["data"] for e in events if e["kind"] == "phi_handwriting"]
    assert skips == [{"status": "skipped", "reason": reason, "model_id": "hw_reader"}]
    assert not [e for e in events if e["kind"] == "model_call"]
    assert on["phi"] == off["phi"] and on["flags"] == off["flags"]
    assert [p.data["text"] for p in on["phi_masked_pages"]] == [p.data["text"] for p in off["phi_masked_pages"]]
    for rel in on["phi"]["masked_artifacts"]:
        assert (on_dir / rel).read_bytes() == (off_dir / rel).read_bytes()


def test_reader_failing_mid_case_records_skip_and_never_fails_the_stage(tmp_path, monkeypatch):
    page, words = _page()
    monkeypatch.setattr(hw_mod, "availability", lambda model_id: None)

    def boom(self, payload, prompt, **kw):
        raise hw_mod.HandwritingUnavailable("load_failed:OSError")

    monkeypatch.setattr(HandwritingReader, "_invoke", boom)
    out, events, _ = _run_stage(tmp_path, monkeypatch, words, page, "boom")
    ev = [e["data"] for e in events if e["kind"] == "phi_handwriting"]
    assert ev and ev[-1]["status"] == "skipped" and ev[-1]["reason"] == "load_failed:OSError"
    assert not [d for d in out["phi"]["detections"] if d["bbox"][0] > 400 and 180 < d["bbox"][1] < 200]


# --------------------------------------------------------------------------- traces and the guard
def test_traces_hold_counts_and_hashes_never_recognised_text(tmp_path, monkeypatch):
    page, words = _page()
    monkeypatch.setattr(hw_mod, "availability", lambda model_id: None)
    monkeypatch.setattr(HandwritingReader, "_invoke", lambda self, payload, prompt, **kw: NAME)
    out, events, out_dir = _run_stage(tmp_path, monkeypatch, words, page, "trace")
    calls = [e["data"] for e in events if e["kind"] == "model_call"]
    assert calls and {c["model_id"] for c in calls} == {"microsoft/trocr-small-handwritten"}
    assert all(c["output_hash"].startswith("sha256:") for c in calls)
    pages = [e["data"] for e in events if e["kind"] == "phi_handwriting"]
    assert pages and pages[0]["status"] == "ok" and pages[0]["n_crops"] >= 1 and pages[0]["n_detections"] >= 1
    blob = "".join(p.read_text(encoding="utf-8") for p in (out_dir / "_traces").rglob("*") if p.is_file())
    blob += json.dumps(out["phi"])
    for s in ("Brooks", NAME, "L. Brooks", "Broo", "Halvorsen", "Quistgaard"):
        assert s not in blob, s


def test_reader_guard_local_only_and_phi_crops_only(monkeypatch):
    monkeypatch.delenv("RADINTAKE_FORBID_MODEL_CALLS", raising=False)
    monkeypatch.setattr(HandwritingReader, "_invoke", lambda self, payload, prompt, **kw: NAME)
    reader = HandwritingReader(config=load_config())
    assert reader.kind == "local"
    img = np.zeros((10, 10), np.uint8)
    with pytest.raises(UnmaskedPayloadError):
        reader.call(MaskedPayload(data={"image": img}, phi_masked=True), prompt_id="p")
    with pytest.raises(UnmaskedPayloadError):
        reader.read(PhiCrop(img))  # not stamped by the PHI agent
    assert reader.read(PhiCrop(img, phi_crop=True)) == NAME
    api = HandwritingReader("remote-hw", kind="api", config=load_config())
    with pytest.raises(NonLocalPhiReader):
        api.read(PhiCrop(img, phi_crop=True))
