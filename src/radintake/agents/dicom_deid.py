"""DICOM de-identification agent: header profile actions + burned-in text redaction.

Header (``configs/dicom_profile.yaml``): PS3.15 Basic Application Confidentiality Profile
actions per attribute keyword - remove (X), zero-length (Z), dummy (D), consistent UID remap
(U), keep (K) - then fail-safe rules: every other person-name (PN) element is emptied, every
other non-registered UID is remapped, private tags are dropped unless safe-listed; sequences
are processed recursively. ``PatientIdentityRemoved=YES``, ``DeidentificationMethod`` and the
method code sequence (113100, plus 113101 Clean Pixel Data when pixels were redacted) are set.

Burned-in text (Clean Pixel Data): per frame the stored values are display-normalised
(:func:`radintake.tools.dicom_io.to_display`; min-max when the VOI window flattens the frame).
Text candidates come from the image itself - bright pixels (``threshold`` x max) kept by a
morphological white top-hat, split into glyph-sized components and grouped into lines - not
only from OCR word boxes. A line is accepted when it sits on a flat field and OCR verifies it
(full-frame pass or line crop; 1-3 glyph tokens such as ``[M]`` by a stroke/aspect check plus a
one-character read), and is kept on the image only when every confident word is in the
``safe_text`` keep list (positional / laterality markers). Accepted lines (merged per text
block) are padded by ``pad_px`` (<= 15) and filled with the value that displays as black.
Every pixel outside those boxes keeps its stored value bit for
bit; the frames are re-encoded with the original transfer syntax when it is lossless and an
encoder exists, otherwise Explicit VR Little Endian (file meta updated).
``BurnedInAnnotation`` becomes ``NO`` only when text was found and redacted, or the source
already declared none; otherwise it stays as-is and the case gets ``phi_uncertain``.

Validity: pydicom round trip (header + pixels) and ``dciodvfy`` when installed.
No header VALUES are ever logged, traced or written to the output JSON - tag keywords only.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from radintake.config import config_dir
from radintake.core.types import BBox
from radintake.logging import register_phi
from radintake.orchestrator.stages import DICOM_TYPES, StageContext, register_stage
from radintake.tools import dicom_io

DEID_VERSION = "dicom-deid-2"
_REGISTERED_ROOT = "1.2.840.10008."
_DUMMY = {"PN": "ANONYMOUS", "LO": "ANONYMIZED", "SH": "ANON", "LT": "ANONYMIZED", "ST": "ANONYMIZED",
          "UT": "ANONYMIZED", "DA": "19000101", "TM": "000000", "DT": "19000101000000", "AS": "000Y",
          "IS": "0", "DS": "0", "CS": "ANON"}
_BUILTIN: dict[str, Any] = {
    "actions": {}, "private_safe_list": [],
    "deidentification_method": "radintake dicom_deid: PS3.15 Basic Application Confidentiality Profile",
    "burned_in": {"pad_px": 4, "threshold": 0.9, "min_conf": 0.3, "ocr_scale": 2},
}
_MAX_PAD = 15


def load_profile(cfg: Any = None) -> dict[str, Any]:
    """Built-in defaults < ``configs/dicom_profile.yaml`` < ``dicom_deid:`` in the run config."""
    p = config_dir() / "dicom_profile.yaml"
    data = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.is_file() else {}
    out = {**_BUILTIN, **(data if isinstance(data, dict) else {})}
    raw = getattr(cfg, "raw", None) or {}
    user = raw.get("dicom_deid") if isinstance(raw.get("dicom_deid"), dict) else {}
    for k, v in user.items():
        out[k] = {**out.get(k, {}), **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    if isinstance(out.get("extra_actions"), dict):  # fail-safe additions found by evaluation
        out["actions"] = {**(out.get("actions") or {}), **out["extra_actions"]}
    bi = dict(out.get("burned_in") or {})
    bi["pad_px"] = min(float(bi.get("pad_px", 4)), _MAX_PAD)
    out["burned_in"] = bi
    return out


# --------------------------------------------------------------------------- UIDs
class UidRemapper:
    """Consistent UID remap: same input UID -> same ``2.25.<int>`` UID for this remapper.

    The salt is secret and random per process by default, so the mapping is consistent across
    every instance of a study in one batch / service process but not reversible offline.
    """

    def __init__(self, salt: bytes | None = None):
        self._salt = salt if salt is not None else secrets.token_bytes(16)
        self._lock = threading.Lock()
        self._map: dict[str, str] = {}

    def __call__(self, uid: str) -> str:
        uid = str(uid).strip()
        if not uid:
            return uid
        with self._lock:
            new = self._map.get(uid)
            if new is None:
                h = hashlib.sha256(self._salt + uid.encode("ascii", "ignore")).digest()
                new = "2.25." + str(int.from_bytes(h[:16], "big"))
                self._map[uid] = new
            return new


_default_remapper = UidRemapper()


def default_remapper() -> UidRemapper:
    return _default_remapper


# --------------------------------------------------------------------------- header
def _register_value(elem) -> None:
    if elem.VR in ("SQ", "UI", "OB", "OW", "OF", "OD", "UN") or elem.value is None:
        return
    vals = elem.value if isinstance(elem.value, (list, tuple)) or type(elem.value).__name__ == "MultiValue" \
        else [elem.value]
    for v in vals:
        s = str(v)
        register_phi(s)
        if elem.VR == "PN":
            register_phi(*[p for p in re.split(r"[\^=\s]+", s) if len(p) >= 3])


def _tag_name(elem) -> str:
    return elem.keyword or f"({elem.tag.group:04X},{elem.tag.element:04X})"


def _private_safe(ds, elem, safe: list[dict[str, Any]]) -> bool:
    if not safe:
        return False
    g, e = elem.tag.group, elem.tag.element
    if 0x0010 <= e <= 0x00FF:  # a private creator element: keep if any safe entry uses it
        return any(str(s.get("creator")) == str(elem.value) for s in safe)
    creator_tag = (g << 16) | (e >> 8)
    creator = ds.get(creator_tag)
    if creator is None:
        return False
    off = e & 0xFF
    return any(str(s.get("creator")) == str(creator.value) and int(str(s.get("element")), 16) == off
               for s in safe)


def deidentify_header(ds, profile: dict[str, Any], remap: UidRemapper,
                      changed: set[str] | None = None) -> set[str]:
    """Apply the profile to ``ds`` in place (recursing into sequences). Returns tag names touched."""
    changed = set() if changed is None else changed
    actions: dict[str, str] = {k: str(v).upper() for k, v in (profile.get("actions") or {}).items()}
    safe = list(profile.get("private_safe_list") or [])
    for elem in list(ds):
        tag = elem.tag
        if tag.group == 0x0002 or tag == 0x7FE00010:  # file meta / pixel data handled elsewhere
            continue
        if tag.is_private:
            if not _private_safe(ds, elem, safe):
                _register_value(elem)
                del ds[tag]
                changed.add(_tag_name(elem))
            continue
        kw = elem.keyword
        act = actions.get(kw)
        if act is None and 0x6000 <= tag.group <= 0x601E and tag.group % 2 == 0:
            act = "X"  # overlay planes (60xx): graphic annotations can carry burned-in-style text
        if act is None:  # fail-safe rules
            if elem.VR == "PN":
                act = "Z"
            elif elem.VR == "UI" and elem.value and not str(elem.value).startswith(_REGISTERED_ROOT):
                act = "U"
            elif elem.VR in ("DA", "DT") and elem.value:
                act = "X"  # any date not named in the profile (HIPAA Safe Harbor: all dates are PHI)
            elif kw and any(kw.startswith(p) for p in profile.get("remove_keyword_prefixes") or ()):
                act = "X"
            else:
                act = "K"
        if act == "K":
            if elem.VR == "SQ":
                for item in elem.value:
                    deidentify_header(item, profile, remap, changed)
            continue
        _register_value(elem)
        if act == "X":
            del ds[tag]
        elif act == "Z":
            elem.value = [] if elem.VR == "SQ" else ""
        elif act == "D":
            elem.value = [] if elem.VR == "SQ" else _DUMMY.get(elem.VR, "")
        elif act == "U":
            if elem.VM > 1:
                elem.value = [remap(str(v)) for v in elem.value]
            else:
                elem.value = remap(str(elem.value))
        else:
            raise ValueError(f"unknown de-id action {act!r} for {kw}")
        changed.add(_tag_name(elem))
    return changed


def _code_item(value: str, meaning: str):
    from pydicom.dataset import Dataset

    it = Dataset()
    it.CodeValue, it.CodingSchemeDesignator, it.CodeMeaning = value, "DCM", meaning
    return it


def _stamp(ds, profile: dict[str, Any], pixels_cleaned: bool) -> None:
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = str(profile.get("deidentification_method") or "")[:64]
    codes = [_code_item("113100", "Basic Application Confidentiality Profile")]
    if pixels_cleaned:
        codes.append(_code_item("113101", "Clean Pixel Data Option"))
    ds.DeidentificationMethodCodeSequence = codes
    ds.LongitudinalTemporalInformationModified = "MODIFIED"


# --------------------------------------------------------------------------- burned-in text
OcrFn = Callable[..., list]


def _default_ocr(img: np.ndarray, **kw: Any) -> list:
    from radintake.tools.ocr import ocr

    return ocr(img, **kw)


def _conf(w: Any) -> float:
    return float(getattr(w, "conf", getattr(w, "confidence", 0.0)))


def _norm(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "", str(text)).upper().strip("-")


def _n_alnum(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]", str(text)))


@dataclass
class _Line:
    """A candidate text line: bright character-sized components side by side (frame coords)."""

    x0: int
    y0: int
    x1: int
    y1: int
    comps: list[tuple[int, int, int, int, int]] = field(default_factory=list)  # x0, y0, x1, y1, area
    words: list[Any] = field(default_factory=list)

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    def add(self, c: tuple[int, int, int, int, int]) -> None:
        self.comps.append(c)
        self.x0, self.y0 = min(self.x0, c[0]), min(self.y0, c[1])
        self.x1, self.y1 = max(self.x1, c[2]), max(self.y1, c[3])


def _components(text_mask: np.ndarray, params: dict[str, Any]) -> list[tuple[int, int, int, int, int]]:
    """Character-sized connected components of the text mask: (x0, y0, x1, y1, area)."""
    import cv2

    n, _lab, st, _c = cv2.connectedComponentsWithStats(text_mask.astype(np.uint8), connectivity=8)
    min_h = int(params.get("min_char_px", 6))
    max_h = max(40.0, float(params.get("max_char_frac", 0.05)) * text_mask.shape[0])
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if min_h <= h <= max_h and w <= 15 * h and a >= 0.08 * w * h:
            out.append((x, y, x + w, y + h, a))
    return out


def _group_lines(comps: list[tuple[int, int, int, int, int]]) -> list[_Line]:
    """Join components that share a baseline band and sit within one character height."""
    lines: list[_Line] = []
    for c in sorted(comps):
        ch = c[3] - c[1]
        for ln in lines:
            ov = min(c[3], ln.y1) - max(c[1], ln.y0)
            if (ov >= 0.5 * min(ch, ln.h) and max(ch, ln.h) <= 1.5 * min(ch, ln.h)
                    and c[0] - ln.x1 <= 1.2 * max(ch, ln.h)):
                ln.add(c)
                break
        else:
            lines.append(_Line(c[0], c[1], c[2], c[3], [c]))
    return lines


def _clean_background(display: np.ndarray, text_mask: np.ndarray, ln: _Line, params: dict[str, Any]) -> bool:
    """Burned-in text sits on a flat field: the ring around the line (text pixels aside) is either
    mostly dark (``bg_max`` x max for ``bg_clean`` of it) or flat (10-90 percentile spread
    <= ``flat_range`` x max) and at least ``min_contrast`` x max darker than the text."""
    hh, ww = display.shape[:2]
    mx = float(display.max())
    m = max(2, round(0.5 * ln.h))
    x0, y0, x1, y1 = max(0, ln.x0 - m), max(0, ln.y0 - m), min(ww, ln.x1 + m), min(hh, ln.y1 + m)
    inner = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    inner[ln.y0 - y0:ln.y1 - y0, ln.x0 - x0:ln.x1 - x0] = True
    sel = ~inner & ~text_mask[y0:y1, x0:x1]
    ring_text = float((~inner & text_mask[y0:y1, x0:x1]).sum())
    bg = display[y0:y1, x0:x1][sel].astype(np.float64)
    if not bg.size:
        return False
    if (float((bg < float(params.get("bg_max", 0.5)) * mx).sum()) + ring_text) / (bg.size + ring_text)             >= float(params.get("bg_clean", 0.75)):
        return True
    p10, p50, p90 = np.percentile(bg, (10, 50, 90))
    return bool(p90 - p10 <= float(params.get("flat_range", 0.25)) * mx
                and p50 <= (1.0 - float(params.get("min_contrast", 0.3))) * mx)


def _short_token_shape(ln: _Line) -> bool:
    """Stroke / aspect heuristic for 1-3 glyph tokens such as ``[M]``, ``L``, ``R``: upright glyph
    boxes made of strokes (neither a solid blob nor a hairline)."""
    if not 1 <= len(ln.comps) <= 3:
        return False
    for x0, y0, x1, y1, a in ln.comps:
        w, h = x1 - x0, y1 - y0
        if not (0.5 <= h / max(w, 1) <= 8.0 and 0.1 <= a / float(w * h) <= 0.8):
            return False
    return True


def _crop_ocr(bright: np.ndarray, ln: _Line, params: dict[str, Any], ocr_fn: OcrFn, trace: Any,
              psm: int) -> list:
    import cv2

    hh, ww = bright.shape[:2]
    m = max(2, ln.h // 3)
    crop = bright[max(0, ln.y0 - m):min(hh, ln.y1 + m), max(0, ln.x0 - m):min(ww, ln.x1 + m)]
    img = np.where(crop, 0, 255).astype(np.uint8)
    f = int(min(8, max(int(params.get("ocr_scale", 2)), int(np.ceil(40.0 / max(ln.h, 1))))))
    if f > 1:
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
    return [w for w in ocr_fn(img, psm=psm, trace=trace) if _ALNUM_OR_BRACKET.search(w.text)]


_ALNUM_OR_BRACKET = re.compile(r"[A-Za-z0-9\[\]]")


def _merge_blocks(boxes: list[BBox], gap: float) -> list[BBox]:
    """Union vertically adjacent, horizontally overlapping line boxes into text blocks."""
    out = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                h = max(a.height, b.height)
                vgap = max(a.y0, b.y0) - min(a.y1, b.y1)
                if vgap <= gap * h and min(a.x1, b.x1) > max(a.x0, b.x0):
                    out[i] = BBox(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))
                    del out[j]
                    changed = True
                    break
            if changed:
                break
    return out


def find_text_boxes(display: np.ndarray, params: dict[str, Any], *, ocr_fn: OcrFn | None = None,
                    trace: Any = None) -> list[BBox]:
    """Boxes (frame pixel coords) of burned-in text on a ``uint8`` display frame.

    1. Candidates independent of OCR word boxes: bright pixels (``threshold`` x max) that a
       morphological white top-hat (``tophat_px`` kernel) keeps - thin strokes, not broad bright
       anatomy - split into character-sized components and grouped into text lines.
    2. Every line must sit on a flat dark field (``bg_max`` / ``bg_clean``) and be verified: a
       confident word (>= 2 characters, ``verify_conf``) from a full-frame OCR pass or from OCR
       of the line crop, or - for 1-3 glyph tokens such as ``[M]`` - the stroke/aspect heuristic
       plus an OCR read of at least one character (``short_conf``).
    3. A verified line is kept on the image only when every confident word on it is in
       ``safe_text`` (positional / laterality markers TCIA keeps); otherwise its tight box is
       redacted. Redacted lines of one text block are merged (``merge_gap`` x line height).
    """
    import cv2

    ocr_fn = ocr_fn or _default_ocr
    mx = int(display.max())
    if mx == 0:
        return []
    bright = display >= float(params.get("threshold", 0.9)) * mx
    if not bright.any() or bright.mean() > 0.5:  # nothing bright, or a mostly-white frame: not text
        return []
    k = max(3, int(params.get("tophat_px", 21)) | 1)
    tophat = cv2.morphologyEx(display, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    text_mask = bright & (tophat >= float(params.get("tophat_min", 0.35)) * mx)
    min_conf = float(params.get("min_conf", 0.3))
    verify_conf = float(params.get("verify_conf", 0.6))
    short_conf = float(params.get("short_conf", 0.5))
    safe = {_norm(s) for s in params.get("safe_text") or ()}

    # full-frame OCR pass (Tesseract sparse text) on the binarised bright pixels
    scale = max(1, int(params.get("ocr_scale", 2)))
    inv = np.where(bright, 0, 255).astype(np.uint8)
    if scale > 1:
        inv = cv2.resize(inv, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    frame_words = []
    for w in ocr_fn(inv, psm=11, trace=trace):
        if _ALNUM_OR_BRACKET.search(w.text):
            b = w.bbox
            frame_words.append((w, BBox(b.x0 / scale, b.y0 / scale, b.x1 / scale, b.y1 / scale)))

    lines = _group_lines(_components(text_mask, params))
    # a confident OCR word that no candidate line covers (e.g. a glyph the top-hat split) seeds its own
    for w, wb in frame_words:
        if _n_alnum(w.text) < 2 or _conf(w) < verify_conf:
            continue
        cx, cy = (wb.x0 + wb.x1) / 2, (wb.y0 + wb.y1) / 2
        if any(ln.x0 <= cx <= ln.x1 and ln.y0 <= cy <= ln.y1 for ln in lines):
            continue
        x0, y0, x1, y1 = int(np.floor(wb.x0)), int(np.floor(wb.y0)), int(np.ceil(wb.x1)), int(np.ceil(wb.y1))
        ys, xs = np.nonzero(bright[y0:y1, x0:x1])
        if len(xs):
            lines.append(_Line(x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1,
                               y0 + int(ys.max()) + 1))
    for w, wb in frame_words:
        cx, cy = (wb.x0 + wb.x1) / 2, (wb.y0 + wb.y1) / 2
        for ln in lines:
            pad = 0.25 * ln.h
            if ln.x0 - pad <= cx <= ln.x1 + pad and ln.y0 - pad <= cy <= ln.y1 + pad:
                ln.words.append(w)
                break

    budget = int(params.get("max_line_ocr", 60))
    boxes: list[BBox] = []
    for ln in sorted(lines, key=lambda ln: (ln.y0, ln.x0)):
        if not _clean_background(display, text_mask, ln, params):
            continue
        words = list(ln.words)
        verified = any(_n_alnum(w.text) >= 2 and _conf(w) >= verify_conf for w in words)
        short = _short_token_shape(ln)
        if not verified and budget > 0:
            budget -= 1
            words += _crop_ocr(bright, ln, params, ocr_fn, trace, psm=7)
            verified = any(_n_alnum(w.text) >= 2 and _conf(w) >= verify_conf for w in words)
            if not verified and short and not words and budget > 0:
                budget -= 1
                words += _crop_ocr(bright, ln, params, ocr_fn, trace, psm=10)
        if not verified and short:
            verified = any(_ALNUM_OR_BRACKET.search(w.text) and _conf(w) >= short_conf for w in words)
        if not verified:
            continue
        # single-glyph fragments below verify_conf (an OCR split of a marker word) do not count
        confident = [w for w in words if _conf(w) >= min_conf and _n_alnum(w.text)
                     and (_n_alnum(w.text) >= 2 or _conf(w) >= verify_conf)]
        if confident and all(_norm(w.text) in safe for w in confident):
            continue  # positional / laterality marker line: TCIA keeps it
        boxes.append(BBox(float(ln.x0), float(ln.y0), float(ln.x1), float(ln.y1)))
    if params.get("merge_lines", True):
        boxes = _merge_blocks(boxes, float(params.get("merge_gap", 0.8)))
    return boxes


def _fill(frame: np.ndarray, boxes: list[BBox], pad: float, value: Any) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    for b in boxes:
        x0, y0 = max(0, int(np.floor(b.x0 - pad))), max(0, int(np.floor(b.y0 - pad)))
        x1, y1 = min(w, int(np.ceil(b.x1 + pad))), min(h, int(np.ceil(b.y1 + pad)))
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = value
    return out


def _display(frame: np.ndarray, ds) -> np.ndarray:
    """Display view for text search; a VOI window that flattens the frame falls back to min-max."""
    disp = dicom_io.to_display(frame, ds)
    if frame.ndim == 2 and disp.max() == disp.min() and frame.max() > frame.min():
        a = frame.astype(np.float64)
        disp = ((a - a.min()) / (a.max() - a.min()) * 255.0).round().astype(np.uint8)
        if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
            disp = 255 - disp
    return disp


def redact_frames(arr: np.ndarray, ds, params: dict[str, Any], *, ocr_fn: OcrFn | None = None,
                  trace: Any = None) -> tuple[np.ndarray, list[tuple[int, BBox]]]:
    """Return (redacted copy of ``arr``, [(frame index 0-based, padded box)])."""
    pad = min(float(params.get("pad_px", 4)), _MAX_PAD)
    out = arr.copy()
    regions: list[tuple[int, BBox]] = []
    for i in range(arr.shape[0]):
        frame = arr[i]
        boxes = find_text_boxes(_display(frame, ds), params, ocr_fn=ocr_fn, trace=trace)
        if not boxes:
            continue
        out[i] = _fill(frame, boxes, pad, dicom_io.black_value(frame, ds))
        h, w = frame.shape[:2]
        regions += [(i, BBox(max(0.0, b.x0 - pad), max(0.0, b.y0 - pad), min(float(w), b.x1 + pad),
                             min(float(h), b.y1 + pad))) for b in boxes]
    return out, regions


# --------------------------------------------------------------------------- file level
@dataclass
class DeidResult:
    output: Path
    valid: bool
    tags: list[str]
    regions: list[tuple[int, BBox]]
    transfer_syntax: str
    reencoded: bool
    uncertain: bool
    notes: list[str] = field(default_factory=list)
    n_frames: int = 0
    validation: dict[str, Any] = field(default_factory=dict)


def deidentify_file(src: str | Path, dst: str | Path, *, profile: dict[str, Any] | None = None,
                    remap: UidRemapper | None = None, ocr_fn: OcrFn | None = None,
                    trace: Any = None) -> DeidResult:
    """De-identify ``src`` into ``dst`` (``src`` is never written)."""
    prof = profile or load_profile()
    remap = remap or default_remapper()
    ds = dicom_io.read(src)
    burned_flag = str(getattr(ds, "BurnedInAnnotation", "") or "").upper()
    tags = deidentify_header(ds, prof, remap)
    if "MediaStorageSOPInstanceUID" in ds.file_meta:
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

    regions: list[tuple[int, BBox]] = []
    notes: list[str] = []
    uncertain = False
    nfr = 0
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if "PixelData" in ds:
        arr = dicom_io.frames(ds)
        nfr = int(arr.shape[0])
        red, regions = redact_frames(arr, ds, prof.get("burned_in") or {}, ocr_fn=ocr_fn, trace=trace)
        if regions or burned_flag == "NO":
            ds.BurnedInAnnotation = "NO"
        elif burned_flag == "YES" or not burned_flag:
            uncertain = True  # declared (or unknown) burned-in text that we could not locate
        _stamp(ds, prof, bool(regions))
        wr = dicom_io.write_frames(ds, red, dst)
        ts, reenc, notes = wr.transfer_syntax, wr.reencoded, wr.notes
        expected = tuple(red.shape)
    else:
        import pydicom

        _stamp(ds, prof, False)
        pydicom.dcmwrite(str(dst), ds, enforce_file_format=True)
        ts, reenc, expected = dicom_io.transfer_syntax(ds), False, None
    v = dicom_io.validate(dst, expected_shape=expected)
    return DeidResult(Path(dst), v.valid, sorted(tags), regions, ts, reenc, uncertain, notes, nfr,
                      {"checks": v.checks, "dciodvfy": v.dciodvfy})


@register_stage("dicom_deid", applies_to=DICOM_TYPES, order=30, failure_flag="phi_uncertain")
def dicom_deid_stage(ctx: StageContext) -> dict[str, Any]:
    t0 = time.perf_counter()
    rel = f"{ctx.input_name}.deid.dcm"
    res = deidentify_file(ctx.path, Path(ctx.out_dir) / rel, profile=load_profile(ctx.config),
                          trace=ctx.trace)
    boxes = [[round(v, 2) for v in b.as_list()] for _, b in res.regions]
    detections = [{"type": "BURNED_IN_TEXT", "page": i + 1, "bbox": [round(v, 2) for v in b.as_list()]}
                  for i, b in res.regions]
    detections += [{"type": "DICOM_HEADER", "page": 1, "dicom_tag": t} for t in res.tags]
    ctx.trace.event("dicom_deid", n_tags=len(res.tags), n_regions=len(res.regions), n_frames=res.n_frames,
                    valid=res.valid, reencoded=res.reencoded, version=DEID_VERSION,
                    dciodvfy=res.validation.get("dciodvfy"), notes=res.notes, fields=res.tags,
                    duration_ms=round((time.perf_counter() - t0) * 1000.0, 3))
    flags = ["phi_uncertain"] if res.uncertain or not res.valid else []
    return {"dicom": {"output_path": rel, "valid": res.valid, "header_tags_deidentified": res.tags,
                      "burned_in_regions": boxes},
            "phi": {"masked_artifacts": [rel], "detections": detections},
            "n_pages": res.n_frames, "flags": flags}
