"""DICOM de-identification agent: header profile actions + burned-in text redaction.

Header (``configs/dicom_profile.yaml``): PS3.15 Basic Application Confidentiality Profile
actions per attribute keyword - remove (X), zero-length (Z), dummy (D), consistent UID remap
(U), keep (K) - then fail-safe rules: every other person-name (PN) element is emptied, every
other non-registered UID is remapped, private tags are dropped unless safe-listed; sequences
are processed recursively. ``PatientIdentityRemoved=YES``, ``DeidentificationMethod`` and the
method code sequence (113100, plus 113101 Clean Pixel Data when pixels were redacted) are set.

Burned-in text (Clean Pixel Data): per frame the stored values are display-normalised
(:func:`radintake.tools.dicom_io.to_display`), the brightest pixels (``threshold`` x max -
burned-in annotations are drawn at or near full intensity) are binarised and OCR'd (lane P1
OCR tool). Only regions where OCR reads real characters are accepted; each accepted region is
grown to the bright text pixels of its line, padded by ``pad_px`` (<= 15) and filled with the
value that displays as black. Every pixel outside those boxes keeps its stored value bit for
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

DEID_VERSION = "dicom-deid-1"
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


def find_text_boxes(display: np.ndarray, params: dict[str, Any], *, ocr_fn: OcrFn | None = None,
                    trace: Any = None) -> list[BBox]:
    """Boxes (frame pixel coords) of burned-in text on a ``uint8`` display frame."""
    import cv2

    ocr_fn = ocr_fn or _default_ocr
    mx = int(display.max())
    if mx == 0:
        return []
    mask = display >= float(params.get("threshold", 0.9)) * mx
    if not mask.any() or mask.mean() > 0.5:  # nothing bright, or a mostly-white frame: not text
        return []
    scale = max(1, int(params.get("ocr_scale", 2)))
    inv = np.where(mask, 0, 255).astype(np.uint8)
    if scale > 1:
        inv = cv2.resize(inv, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    min_conf = float(params.get("min_conf", 0.3))
    safe = {s.upper() for s in params.get("safe_text") or ()}
    raw = [w for w in ocr_fn(inv, psm=11, trace=trace) if re.search(r"[A-Za-z0-9]", w.text)]

    def conf(w) -> float:
        return float(getattr(w, "conf", getattr(w, "confidence", 0.0)))

    def is_safe(w) -> bool:
        return re.sub(r"[^A-Za-z0-9-]", "", w.text).upper() in safe

    # group words into text lines (vertical overlap); a line is redacted as a whole - single letters
    # such as "[M]" between a name and a date included - unless every confident word on it is a
    # known-safe radiographic marker (laterality, view, position). Unknown text is always redacted.
    lines: list[list[Any]] = []
    for w in sorted(raw, key=lambda w: (w.bbox.y0, w.bbox.x0)):
        for ln in lines:
            r = ln[-1].bbox
            if min(w.bbox.y1, r.y1) - max(w.bbox.y0, r.y0) > 0.5 * min(w.bbox.height, r.height):
                ln.append(w)
                break
        else:
            lines.append([w])
    words = []
    for ln in lines:
        strong = [w for w in ln if conf(w) >= min_conf and len(re.findall(r"[A-Za-z0-9]", w.text)) >= 2]
        if not strong or all(is_safe(w) for w in ln if conf(w) >= min_conf):
            continue
        words += [w for w in ln if not is_safe(w)]
    if not words:
        return []
    # group bright pixels into text lines; grow each accepted word to its line's bright pixels
    _n, labels, _stats, _c = cv2.connectedComponentsWithStats(
        cv2.dilate(mask.astype(np.uint8), np.ones((3, 15), np.uint8)), connectivity=8)
    boxes: list[BBox] = []
    for w in words:
        b = w.bbox
        wb = BBox(b.x0 / scale, b.y0 / scale, b.x1 / scale, b.y1 / scale)
        x0, y0, x1, y1 = (int(np.floor(wb.x0)), int(np.floor(wb.y0)), int(np.ceil(wb.x1)), int(np.ceil(wb.y1)))
        ids = {int(i) for i in np.unique(labels[y0:y1, x0:x1]) if i != 0}
        best = wb
        for i in ids:
            ys, xs = np.nonzero((labels == i) & mask)
            if not len(xs):
                continue
            cb = BBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            if cb.area <= 4 * max(wb.area, 1.0):  # never grow into a large bright structure
                best = BBox(min(best.x0, cb.x0), min(best.y0, cb.y0), max(best.x1, cb.x1), max(best.y1, cb.y1))
        if best not in boxes:  # words of one line grow to the same box
            boxes.append(best)
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


def redact_frames(arr: np.ndarray, ds, params: dict[str, Any], *, ocr_fn: OcrFn | None = None,
                  trace: Any = None) -> tuple[np.ndarray, list[tuple[int, BBox]]]:
    """Return (redacted copy of ``arr``, [(frame index 0-based, padded box)])."""
    pad = min(float(params.get("pad_px", 4)), _MAX_PAD)
    out = arr.copy()
    regions: list[tuple[int, BBox]] = []
    for i in range(arr.shape[0]):
        frame = arr[i]
        boxes = find_text_boxes(dicom_io.to_display(frame, ds), params, ocr_fn=ocr_fn, trace=trace)
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
