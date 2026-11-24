"""Input router - a FIXED pipeline step, deliberately not an agent.

Classifying an input file is deterministic, cheap and fully testable with magic bytes and a
few image statistics; an LLM would add cost, latency and non-determinism and nothing else.

Order of checks (magic bytes first, extension second):

1. ``DICM`` at offset 128 -> ``dicom`` (header must parse);
2. ``%PDF`` -> ``pdf`` (must open, >= 1 page);
3. ``II*\\0`` / ``MM\\0*`` -> ``tiff`` (all frames must decode);
4. PNG / JPEG (or anything Pillow decodes) -> ``scan`` or ``photo`` by a quality heuristic;
5. no magic match -> preamble-less DICOM attempt (``pydicom`` with ``force=True``), accepted only
   when real DICOM UIDs are present;
6. anything else, or a file that fails to decode -> ``unknown`` with a non-empty ``error``.

``route`` never raises: every failure becomes ``unknown`` + an error reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROUTER_VERSION = "router.v1"
_HEAD = 4096
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
_CAMERA_EXIF = (0x010F, 0x0110)  # Make, Model

# scan-vs-photo thresholds (tuned on the synthetic tiers; see test_router)
_BG_RANGE_PHOTO = 0.12   # relative spread of the estimated paper background (uneven lighting)
_BORDER_DARK_PHOTO = 0.30  # fraction of dark pixels in the outer margin (desk around the page)


@dataclass
class RouteResult:
    input_type: str  # dicom | pdf | tiff | scan | photo | unknown
    error: str | None = None
    n_pages: int | None = None
    signals: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.input_type != "unknown"


def _unknown(reason: str) -> RouteResult:
    return RouteResult("unknown", error=reason or "unrecognised input")


_PATHLIKE = re.compile(r"""(['"])([^'"]*[\\/][^'"]*)\1""")


def _err(exc: BaseException) -> str:
    """Short error reason; quoted filesystem paths are cut to their base name."""
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    msg = _PATHLIKE.sub(lambda m: m.group(1) + re.split(r"[\\/]+", m.group(2))[-1] + m.group(1), msg)[:200]
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


# --- per-format validators ---------------------------------------------------------------
def _check_dicom(path: Path, force: bool) -> RouteResult:
    import pydicom

    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=force)
    if force:
        uids = [getattr(ds, k, None) for k in ("SOPClassUID", "SOPInstanceUID", "StudyInstanceUID")]
        tsx = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        if not any(uids) and not tsx:
            return _unknown("not a DICOM file (no DICOM UIDs found)")
    n = int(getattr(ds, "NumberOfFrames", 1) or 1)
    return RouteResult("dicom", n_pages=n)


def _check_pdf(path: Path) -> RouteResult:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        n = len(pdf)
    finally:
        pdf.close()
    if n < 1:
        return _unknown("PDF has no pages")
    return RouteResult("pdf", n_pages=n)


def _check_tiff(path: Path) -> RouteResult:
    from PIL import Image, ImageSequence

    n = 0
    with Image.open(path) as img:
        for frame in ImageSequence.Iterator(img):
            frame.load()
            n += 1
    if n < 1:
        return _unknown("TIFF has no frames")
    return RouteResult("tiff", n_pages=n)


def scan_or_photo(gray: np.ndarray, exif_camera: bool = False) -> tuple[str, dict[str, float]]:
    """Cheap scan-vs-photo heuristic on a grayscale uint8 page image.

    Signals: camera EXIF, unevenness of the paper background (illumination variance after
    removing text with a morphological closing), and dark pixels in the outer margin (desk or
    background around a photographed page).
    """
    import cv2

    h, w = gray.shape[:2]
    scale = 512.0 / max(h, w)
    small = cv2.resize(gray, (max(8, int(w * scale)), max(8, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else gray
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), 5).astype(np.float32)
    p5, p95 = np.percentile(bg, [5, 95])
    bg_range = float((p95 - p5) / max(p95, 1.0))
    sh, sw = small.shape[:2]
    m = max(2, int(0.03 * min(sh, sw)))
    margin = np.concatenate([small[:m].ravel(), small[-m:].ravel(), small[:, :m].ravel(),
                             small[:, -m:].ravel()])
    border_dark = float((margin < 100).mean())
    signals = {"bg_range": round(bg_range, 4), "border_dark": round(border_dark, 4),
               "exif_camera": float(exif_camera), "aspect": round(max(h, w) / max(1, min(h, w)), 4)}
    is_photo = exif_camera or bg_range > _BG_RANGE_PHOTO or border_dark > _BORDER_DARK_PHOTO
    return ("photo" if is_photo else "scan"), signals


def _check_image(path: Path) -> RouteResult:
    from PIL import Image

    with Image.open(path) as img:
        img.load()  # full decode: a truncated / empty file fails here
        exif = img.getexif()
        exif_camera = bool(exif and any(exif.get(t) for t in _CAMERA_EXIF))
        gray = np.asarray(img.convert("L"))
    if gray.size == 0:
        return _unknown("image has no pixels")
    kind, signals = scan_or_photo(gray, exif_camera)
    return RouteResult(kind, n_pages=1, signals=signals)


def _try_forced_dicom(path: Path) -> RouteResult | None:
    """Preamble-less DICOM attempt; ``None`` when the file is not DICOM."""
    try:
        res = _check_dicom(path, force=True)
    except Exception:  # noqa: BLE001 - not DICOM
        return None
    return res if res.ok else None


# --- public API --------------------------------------------------------------------------
def sniff(head: bytes) -> str | None:
    """Magic-byte classification of a file head (no decoding)."""
    if len(head) >= 132 and head[128:132] == b"DICM":
        return "dicom"
    if b"%PDF" in head[:1024]:
        return "pdf"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head[:8] == b"\x89PNG\r\n\x1a\n" or head[:3] == b"\xff\xd8\xff":
        return "image"
    return None


def route(path: str | Path) -> RouteResult:
    """Classify one input file. Never raises."""
    p = Path(path)
    try:
        if not p.is_file():
            return _unknown("not a regular file")
        with p.open("rb") as fh:
            head = fh.read(_HEAD)
        if not head:
            return _unknown("empty file")
        kind = sniff(head)
        ext = p.suffix.lower()
        try:
            if kind == "dicom":
                return _check_dicom(p, force=False)
            if kind == "pdf":
                return _check_pdf(p)
            if kind == "tiff":
                return _check_tiff(p)
            if kind == "image":
                return _check_image(p)
            # no magic match: extension second
            if ext in _IMAGE_EXTS:
                return _check_image(p)
            res = _try_forced_dicom(p)
            if res is not None:
                return res
            return _unknown(f"unsupported input (no known signature, extension {ext or 'none'!r})")
        except Exception as exc:  # noqa: BLE001 - decode failure -> unknown, never crash
            label = kind or ext.lstrip(".") or "file"
            return _unknown(f"failed to decode {label}: {_err(exc)}")
    except Exception as exc:  # noqa: BLE001 - absolute guarantee: route never raises
        return _unknown(f"router failure: {_err(exc)}")
