"""DICOM I/O helpers: decode frames, display-normalise, re-encode, validate.

* :func:`frames` decodes PixelData (compressed transfer syntaxes included, as far as the
  installed pydicom decoders allow; RLE is native) into ``(n_frames, rows, cols[, samples])``.
* :func:`to_display` windows/normalises one frame to ``uint8`` for OCR only - the redaction
  itself is always applied to the RAW stored values.
* :func:`write_frames` stores frames back: re-encoded with the ORIGINAL transfer syntax when
  that syntax is lossless and an encoder is available (so pixels outside redaction boxes stay
  bit-identical), otherwise Explicit VR Little Endian with the file meta updated.
* :func:`validate` = pydicom round-trip read + pixel decode (+ ``dciodvfy`` when on PATH).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Lossless compressed syntaxes we may re-encode into (if pydicom has an encoder for them).
LOSSLESS_COMPRESSED = frozenset({
    "1.2.840.10008.1.2.5",       # RLE Lossless
    "1.2.840.10008.1.2.4.80",    # JPEG-LS Lossless
    "1.2.840.10008.1.2.4.90",    # JPEG 2000 Lossless
    "1.2.840.10008.1.2.4.70",    # JPEG Lossless SV1
    "1.2.840.10008.1.2.4.57",    # JPEG Lossless
})


def read(path: str | Path):
    import pydicom

    return pydicom.dcmread(str(path))


def transfer_syntax(ds) -> str:
    meta = getattr(ds, "file_meta", None)
    return str(getattr(meta, "TransferSyntaxUID", "") or "1.2.840.10008.1.2")


def is_compressed(ds) -> bool:
    from pydicom.uid import UID

    return UID(transfer_syntax(ds)).is_compressed


def n_frames(ds) -> int:
    try:
        return max(1, int(getattr(ds, "NumberOfFrames", 1) or 1))
    except (TypeError, ValueError):
        return 1


def frames(ds) -> np.ndarray:
    """All frames as ``(n, rows, cols[, samples])`` in stored values (YBR left as stored)."""
    arr = ds.pixel_array if not _is_ybr(ds) else _pixel_array_raw(ds)
    if n_frames(ds) == 1:
        arr = arr[None, ...]
    return np.ascontiguousarray(arr)


def _is_ybr(ds) -> bool:
    return str(getattr(ds, "PhotometricInterpretation", "")).startswith("YBR")


def _pixel_array_raw(ds) -> np.ndarray:
    from pydicom.pixels import pixel_array

    return pixel_array(ds, raw=True)


def to_display(frame: np.ndarray, ds) -> np.ndarray:
    """``uint8`` view of one frame for OCR: VOI LUT / window if present, else min-max."""
    a = np.asarray(frame)
    if a.ndim == 3:  # colour -> luminance (good enough to find burned-in text)
        a = a[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    else:
        from pydicom.pixels import apply_voi_lut

        if "WindowCenter" in ds or "VOILUTSequence" in ds:
            try:
                a = apply_voi_lut(a, ds)
            except Exception:  # noqa: BLE001 - a broken VOI LUT must not block redaction
                a = np.asarray(frame)  # fall back to plain min-max of the stored values
    a = a.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    out = np.zeros(a.shape, np.uint8) if hi <= lo else ((a - lo) / (hi - lo) * 255.0).round().astype(np.uint8)
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        out = 255 - out  # display white = high
    return out


def black_value(frame: np.ndarray, ds) -> Any:
    """The stored value that DISPLAYS as black for this frame."""
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        return frame.max() if frame.ndim == 2 else frame.reshape(-1, frame.shape[-1]).max(axis=0)
    return frame.min() if frame.ndim == 2 else frame.reshape(-1, frame.shape[-1]).min(axis=0)


@dataclass
class WriteResult:
    transfer_syntax: str
    reencoded: bool  # True = original compressed syntax kept
    notes: list[str] = field(default_factory=list)


def write_frames(ds, arr: np.ndarray, path: str | Path) -> WriteResult:
    """Store ``arr`` (same shape as :func:`frames`) into ``ds`` and write it to ``path``."""
    import pydicom
    from pydicom.uid import ExplicitVRLittleEndian

    data = arr[0] if n_frames(ds) == 1 else arr
    ts = transfer_syntax(ds)
    compressed = is_compressed(ds)
    notes: list[str] = []
    res: WriteResult | None = None
    if compressed and ts in LOSSLESS_COMPRESSED:
        try:
            ds.compress(ts, data, generate_instance_uid=False)
            res = WriteResult(ts, True)
        except Exception as exc:  # noqa: BLE001 - no encoder installed -> uncompressed fallback
            notes.append(f"reencode_failed:{type(exc).__name__}")
    elif compressed:
        notes.append("lossy_source_stored_uncompressed")  # re-encoding lossy would alter pixels
    if res is None:
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        if str(getattr(ds, "PhotometricInterpretation", "")) == "YBR_FULL_422":
            ds.PhotometricInterpretation = "YBR_FULL"  # decoded samples are full resolution
        ds.PixelData = np.ascontiguousarray(data).tobytes()
        ds["PixelData"].VR = "OW" if int(getattr(ds, "BitsAllocated", 8)) > 8 else "OB"
        if "PlanarConfiguration" in ds and data.ndim >= 3 and data.shape[-1] in (3, 4):
            ds.PlanarConfiguration = 0  # numpy layout is pixel-interleaved
        res = WriteResult(str(ExplicitVRLittleEndian), False, notes)
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    pydicom.dcmwrite(str(path), ds, enforce_file_format=True)
    return res


@dataclass
class Validation:
    valid: bool
    checks: dict[str, bool] = field(default_factory=dict)
    dciodvfy: str = "not_available"  # not_available | ok | errors


def _dciodvfy(path: Path) -> str:
    exe = shutil.which("dciodvfy")
    if not exe:
        return "not_available"
    try:
        out = subprocess.run([exe, str(path)], capture_output=True, text=True, timeout=120, check=False)
    except Exception:  # noqa: BLE001
        return "not_available"
    text = out.stdout + out.stderr
    return "errors" if any(ln.lstrip().startswith("Error") for ln in text.splitlines()) else "ok"


def validate(path: str | Path, *, expected_shape: tuple[int, ...] | None = None) -> Validation:
    """pydicom round trip (header + pixels) and, when installed, dciodvfy."""
    p = Path(path)
    checks: dict[str, bool] = {}
    try:
        ds = read(p)
        checks["read"] = True
        checks["sop_uids"] = bool(getattr(ds, "SOPClassUID", "")) and bool(getattr(ds, "SOPInstanceUID", ""))
        checks["meta_consistent"] = str(ds.file_meta.MediaStorageSOPInstanceUID) == str(ds.SOPInstanceUID)
        if "PixelData" in ds:
            arr = frames(ds)
            checks["pixels"] = arr.size > 0
            if expected_shape is not None:
                checks["shape"] = tuple(arr.shape) == tuple(expected_shape)
    except Exception:  # noqa: BLE001 - any failure = invalid
        checks["read"] = checks.get("read", False)
        checks["exception"] = False
    dv = _dciodvfy(p)
    ok = all(checks.values()) and dv != "errors"
    return Validation(ok, checks, dv)
