"""Document PHI agent: detect PHI on OCR words, mask page images and text, keep a local vault.

Runs after preprocessing and BEFORE extraction or any model call (stage order 30). It is the
only producer of payloads stamped ``phi_masked=True``; the guarded model client
(:mod:`radintake.tools.model_client`) refuses everything else.

Layered detector (``configs/phi.yaml``)
---------------------------------------
1. **Layout anchors** - a label on an OCR line (``Patient Name:``, ``Patient ID``, ``MRN``,
   ``DOB`` / ``Date of Birth``, ``Accession``); the value is the tokens to its right, up to
   the next label or the end of the line. Anchors give the typed header fields.
2. **Regex** - MRN / accession formats, long digit runs, dates in common formats.
3. **Names in free text** - Presidio (spaCy NER) when it imports offline (``ner: auto``);
   otherwise a documented lightweight fallback: a capitalised name after an honorific
   (``Dr.``, ``Mr.``, ``Prof`` ...) or before a credential (``, MD``). Every token of a
   patient name found by an anchor is also masked wherever it re-appears on the page.
   The fallback needs no model download, so the stage works with ``RADINTAKE_OFFLINE=1``.

Masking
-------
* Image: detected word boxes are mapped to ORIGINAL page coordinates, padded by
  ``mask.pad_px`` and filled with a uniform value on a COPY of the original page (the masked
  artifact, ``<out>/<input>.masked/page_N.png``); the same boxes (current coords) are filled
  on a copy of the processed page that becomes the model payload.
* Text: detected spans are replaced with typed placeholders (``[PATIENT_NAME_1]``); the
  :class:`~radintake.agents.phi_vault.PhiVault` maps them back locally and registers every
  value with the log/trace scrubber.

Outputs: ``phi`` (``masked_artifacts`` relative to ``--out``, ``detections`` with type / page /
bbox in original coords), plus in-process only ``phi_vault``, ``phi_masked_pages`` (one
:class:`MaskedPayload` per page, the only objects stamped ``phi_masked=True``) and
``ocr_words`` (page index -> OCR words on the processed page) for the extraction stage.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from radintake.agents.phi_vault import PhiVault
from radintake.config import config_dir
from radintake.core.types import BBox, Page
from radintake.orchestrator.stages import DOCUMENT_TYPES, StageContext, register_stage
from radintake.tools.model_client import MaskedPayload

PHI_VERSION = "phi-docs-1"
HEADER_TYPES = ("PATIENT_NAME", "PATIENT_ID", "DATE_OF_BIRTH", "ACCESSION_NUMBER")

# Overlapping spans keep the type of the most specific layer.
_PRIORITY = {"anchor": 3, "known": 2, "regex": 1, "ner": 1}
_TYPE_RANK = {t: 10 for t in HEADER_TYPES} | {"PERSON": 5, "DATE": 4, "ID": 3}

_BUILTIN: dict[str, Any] = {
    "anchors": [], "regex": [], "ner": "fallback",
    "honorifics": ["dr", "doctor", "prof", "professor", "mr", "mrs", "ms", "miss", "mx"],
    "mask": {"pad_px": 6, "fill": "background"},
}


# --------------------------------------------------------------------------- config
def _read_phi_yaml() -> dict[str, Any]:
    p = config_dir() / "phi.yaml"
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def load_params(cfg: Any = None) -> dict[str, Any]:
    """Built-in defaults < ``configs/phi.yaml`` < the ``phi:`` section of the run config."""
    out = {**_BUILTIN, **_read_phi_yaml()}
    raw = getattr(cfg, "raw", None) or {}
    user = raw.get("phi") if isinstance(raw.get("phi"), dict) else {}
    for k, v in user.items():
        out[k] = {**out.get(k, {}), **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# --------------------------------------------------------------------------- spans
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    type: str
    source: str  # anchor | regex | ner | known


def _merge_spans(spans: Iterable[Span]) -> list[Span]:
    """Union overlapping spans; the merged span keeps the strongest (source, type)."""
    out: list[Span] = []
    for s in sorted(spans, key=lambda s: (s.start, -s.end)):
        if out and s.start < out[-1].end:
            a = out[-1]
            best = max((a, s), key=lambda x: (_PRIORITY.get(x.source, 0), _TYPE_RANK.get(x.type, 0)))
            out[-1] = Span(a.start, max(a.end, s.end), best.type, best.source)
        else:
            out.append(s)
    return out


@lru_cache(maxsize=1)
def _presidio():
    """Presidio analyzer if it (and a spaCy model) loads offline, else None."""
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]

        engine = AnalyzerEngine()
        engine.analyze(text="Dr. John Smith", entities=["PERSON"], language="en")
        return engine
    except Exception:  # noqa: BLE001 - not installed / no model offline -> documented fallback
        return None


class PhiDetector:
    """Text-level layered detector; page detection runs it per OCR line."""

    def __init__(self, params: dict[str, Any] | None = None):
        p = params if params is not None else load_params()
        self.params = p
        self.anchors = [(a["type"].upper(), re.compile(a["pattern"], re.IGNORECASE))
                        for a in p.get("anchors") or []]
        self.regex = [(r["type"].upper(), re.compile(r["pattern"], re.IGNORECASE))
                      for r in p.get("regex") or []]
        hon = "|".join(re.escape(h) for h in p.get("honorifics") or [])
        name = r"[A-Z][a-zA-Z'\-]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-zA-Z'\-]+){0,2}"
        self._honorific = re.compile(rf"\b(?i:{hon})\.?\s+(?P<name>{name})") if hon else None
        # "First Last, MD" / "First M. Last, RT" / "G. Rodriguez, RT" (clinician sign-offs)
        self._credential = re.compile(
            r"(?P<name>\b(?:[A-Z]\.\s*|[A-Z][a-zA-Z'\-]+(?:\s+[A-Z]\.)?\s+)[A-Z][a-zA-Z'\-]+),?\s+"
            r"(?:MD|M\.D\.|DO|PhD|RN|NP|PA-C|RT|R\.T\.|RRT|ARRT|RT\(R\))(?![A-Za-z])")
        mode = str(p.get("ner", "auto")).lower()
        self.ner_backend = "presidio" if mode == "auto" and _presidio() is not None else "fallback"

    # -- layers -----------------------------------------------------------------------
    def anchor_spans(self, text: str) -> list[Span]:
        out: list[Span] = []
        offset = 0
        for line in text.split("\n"):
            labels = sorted(((m.start(), m.end(), t) for t, rx in self.anchors for m in rx.finditer(line)),
                            key=lambda x: (x[0], -x[1]))
            kept: list[tuple[int, int, str]] = []
            for lab in labels:  # drop labels nested in an earlier, longer label
                if kept and lab[0] < kept[-1][1]:
                    continue
                kept.append(lab)
            for i, (_, end, t) in enumerate(kept):
                stop = kept[i + 1][0] if i + 1 < len(kept) else len(line)
                seg = line[end:stop]
                v0 = end + (len(seg) - len(seg.lstrip(" \t:#.")))
                v1 = end + len(seg.rstrip())
                if v1 > v0:
                    out.append(Span(offset + v0, offset + v1, t, "anchor"))
            offset += len(line) + 1
        return out

    def regex_spans(self, text: str) -> list[Span]:
        return [Span(m.start(), m.end(), t, "regex") for t, rx in self.regex for m in rx.finditer(text)]

    def ner_spans(self, text: str) -> list[Span]:
        out: list[Span] = []
        if self.ner_backend == "presidio":
            eng = _presidio()
            res = eng.analyze(text=text, entities=["PERSON"], language="en") if eng else []
            out += [Span(r.start, r.end, "PERSON", "ner") for r in res if r.score >= 0.6]
        # honorific / credential rules always run: NER models miss "G. Rodriguez, RT"-style sign-offs
        for rx in (self._honorific, self._credential):
            if rx is None:
                continue
            out += [Span(m.start("name"), m.end("name"), "PERSON", "ner") for m in rx.finditer(text)]
        return out

    @staticmethod
    def known_spans(text: str, known_tokens: Iterable[str]) -> list[Span]:
        toks = sorted({t for t in known_tokens if len(t) >= 3}, key=len, reverse=True)
        if not toks:
            return []
        rx = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in toks) + r")\b", re.IGNORECASE)
        return [Span(m.start(), m.end(), "PATIENT_NAME", "known") for m in rx.finditer(text)]

    def detect(self, text: str, known_tokens: Iterable[str] = ()) -> list[Span]:
        spans = self.anchor_spans(text) + self.regex_spans(text) + self.ner_spans(text)
        spans += self.known_spans(text, known_tokens)
        return _merge_spans(spans)


def name_tokens(values: Iterable[str]) -> set[str]:
    return {t for v in values for t in re.split(r"[\s,]+", v) if len(t) >= 3}


def mask_text(text: str, vault: PhiVault, detector: PhiDetector | None = None,
              known_tokens: Iterable[str] = ()) -> tuple[str, list[Span]]:
    """Replace PHI spans in free text with typed placeholders recorded in ``vault``."""
    det = detector or PhiDetector()
    spans = det.detect(text, known_tokens)
    # a patient name found by an anchor is also PHI wherever it re-appears in the text
    names = [text[s.start:s.end] for s in spans if s.type == "PATIENT_NAME"]
    extra = name_tokens(names) - set(known_tokens)
    if extra:
        spans = _merge_spans(spans + det.known_spans(text, extra))
    parts, pos = [], 0
    for s in spans:
        parts += [text[pos:s.start], vault.placeholder(s.type, text[s.start:s.end])]
        pos = s.end
    parts.append(text[pos:])
    return "".join(parts), spans


# --------------------------------------------------------------------------- pages
@dataclass
class Detection:
    type: str
    page: int
    bbox: BBox  # ORIGINAL page coordinates (unpadded)
    bbox_current: BBox  # processed-page coordinates
    source: str
    placeholder: str
    confidence: float

    def to_output(self) -> dict[str, Any]:
        return {"type": self.type, "page": self.page,
                "bbox": [round(v, 2) for v in self.bbox.as_list()]}


@dataclass
class PageMaskResult:
    page_index: int
    masked_original: np.ndarray  # original page with PHI boxes filled (the artifact)
    payload: MaskedPayload  # processed page + masked text, stamped phi_masked=True
    detections: list[Detection]
    masked_text: str


def _lines(words: Sequence[Any]) -> list[list[int]]:
    """Group word indexes into lines: Tesseract line ids when present, else y overlap."""
    if words and hasattr(words[0], "line_id"):
        groups: dict[tuple, list[int]] = {}
        for i, w in enumerate(words):
            groups.setdefault(w.line_id, []).append(i)
        return [sorted(g, key=lambda i: words[i].bbox.x0) for g in groups.values()]
    lines: list[list[int]] = []
    for i in sorted(range(len(words)), key=lambda i: (words[i].bbox.y0, words[i].bbox.x0)):
        b = words[i].bbox
        for ln in lines:
            r = words[ln[-1]].bbox
            ov = min(b.y1, r.y1) - max(b.y0, r.y0)
            if ov > 0.5 * min(b.height, r.height):
                ln.append(i)
                break
        else:
            lines.append([i])
    return [sorted(ln, key=lambda i: words[i].bbox.x0) for ln in lines]


def _union(boxes: Sequence[BBox]) -> BBox:
    return BBox(min(b.x0 for b in boxes), min(b.y0 for b in boxes),
                max(b.x1 for b in boxes), max(b.y1 for b in boxes))


def _fill_value(img: np.ndarray, fill: Any) -> Any:
    if isinstance(fill, (int, float)) and not isinstance(fill, bool):
        return int(np.clip(fill, 0, 255))
    if img.ndim == 3:
        return np.percentile(img.reshape(-1, img.shape[2]), 95, axis=0).astype(img.dtype)
    return np.percentile(img, 95).astype(img.dtype)


def mask_image(img: np.ndarray, boxes: Iterable[BBox], *, pad: float = 6, fill: Any = "background") -> np.ndarray:
    """A COPY of ``img`` with every (padded) box filled with one uniform value."""
    out = np.array(img, copy=True)
    h, w = out.shape[:2]
    val = _fill_value(img, fill)
    for b in boxes:
        x0, y0 = int(np.floor(max(0.0, b.x0 - pad))), int(np.floor(max(0.0, b.y0 - pad)))
        x1, y1 = int(np.ceil(min(float(w), b.x1 + pad))), int(np.ceil(min(float(h), b.y1 + pad)))
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = val
    return out


@dataclass
class LineHit:
    """One OCR line: its text, word char offsets and the PHI spans found in it."""

    text: str
    offsets: list[tuple[int, int, int]]  # (start, end, word index) in ``text``
    spans: list[Span]

    def words_of(self, s: Span) -> list[int]:
        return [i for a, b, i in self.offsets if a < s.end and b > s.start]


def detect_words(words: Sequence[Any], detector: PhiDetector,
                 known_tokens: Iterable[str] = ()) -> list[LineHit]:
    """Run the detector per OCR line (reading order)."""
    known = set(known_tokens)
    hits: list[LineHit] = []
    for ln in _lines(words):
        parts, offs, pos = [], [], 0
        for i in ln:
            t = words[i].text
            offs.append((pos, pos + len(t), i))
            parts.append(t)
            pos += len(t) + 1
        hits.append(LineHit(" ".join(parts), offs, []))
    # pass 1 finds anchored patient names, pass 2 masks their tokens anywhere on the page
    for h in hits:
        for s in detector.detect(h.text):
            if s.type == "PATIENT_NAME":
                known |= name_tokens([h.text[s.start:s.end]])
    for h in hits:
        h.spans = [s for s in detector.detect(h.text, known) if h.words_of(s)]
    return hits


def mask_page(page: Page, original: Page | None, words: Sequence[Any], detector: PhiDetector,
              vault: PhiVault, *, pad: float = 6, fill: Any = "background",
              known_tokens: Iterable[str] = ()) -> PageMaskResult:
    """Detect PHI on ``words`` (processed-page coords) and mask both page images and the text."""
    original = original or page
    dets: list[Detection] = []
    masked_words: set[int] = set()
    lines_out: list[str] = []
    for h in detect_words(words, detector, known_tokens):
        parts, pos = [], 0
        for s in h.spans:
            idx = h.words_of(s)
            ph = vault.placeholder(s.type, h.text[s.start:s.end])
            cur = _union([words[i].bbox for i in idx])
            conf = float(min(getattr(words[i], "conf", getattr(words[i], "confidence", 1.0)) for i in idx))
            dets.append(Detection(s.type, page.index, page.to_original(cur), cur, s.source, ph, conf))
            masked_words.update(idx)
            parts += [h.text[pos:s.start], ph]
            pos = s.end
        parts.append(h.text[pos:])
        lines_out.append("".join(parts))
    # mask every detected WORD box (not only the union) so gaps between lines stay intact
    cur_boxes = [words[i].bbox for i in sorted(masked_words)]
    masked_original = mask_image(original.image, [page.to_original(b) for b in cur_boxes], pad=pad, fill=fill)
    masked_current = mask_image(page.image, cur_boxes, pad=pad, fill=fill)
    masked_text = "\n".join(lines_out)
    payload = MaskedPayload(data={"image": masked_current, "text": masked_text, "page": page.index},
                            phi_masked=True,
                            meta={"page": page.index, "n_detections": len(dets), "phi_version": PHI_VERSION})
    return PageMaskResult(page.index, masked_original, payload, dets, masked_text)


def _save_png(img: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.png")
    Image.fromarray(img).save(tmp, format="PNG")
    tmp.replace(path)


@register_stage("phi", applies_to=DOCUMENT_TYPES, order=30, failure_flag="phi_uncertain")
def phi_stage(ctx: StageContext) -> dict[str, Any]:
    from radintake.tools.ocr import ocr

    params = load_params(ctx.config)
    mcfg = params.get("mask") or {}
    pad, fill = float(mcfg.get("pad_px", 6)), mcfg.get("fill", "background")
    detector = PhiDetector(params)
    vault = PhiVault()
    pages: list[Page] = list(ctx.data.get("pages") or [])
    originals = {p.index: p for p in (ctx.data.get("original_pages") or pages)}
    rel_dir = f"{ctx.input_name}.masked"
    t0 = time.perf_counter()

    artifacts: list[str] = []
    detections: list[dict[str, Any]] = []
    payloads: list[MaskedPayload] = []
    ocr_words: dict[int, list[Any]] = {}
    out_pages: list[Page] = []
    flags: set[str] = set()
    known: set[str] = set()
    for pg in pages:
        words = ocr(pg.image, trace=ctx.trace)
        ocr_words[pg.index] = words
        if not words:
            flags.add("phi_uncertain")  # nothing readable: cannot vouch for this page
        res = mask_page(pg, originals.get(pg.index), words, detector, vault, pad=pad, fill=fill,
                        known_tokens=known)
        known |= name_tokens(vault.values_of("PATIENT_NAME"))
        rel = f"{rel_dir}/page_{pg.index}.png"
        _save_png(res.masked_original, Path(ctx.out_dir) / rel)
        artifacts.append(rel)
        detections += [d.to_output() for d in res.detections]
        payloads.append(res.payload)
        out_pages.append(replace(pg, words=[w.to_word(pg.index) if hasattr(w, "to_word") else w
                                            for w in words]))
        counts: dict[str, int] = {}
        for d in res.detections:
            counts[d.type] = counts.get(d.type, 0) + 1
        ctx.trace.event("phi_page", page=pg.index, n_words=len(words), n_detections=len(res.detections),
                        counts=counts, backend=detector.ner_backend)
    if pages and not detections:
        flags.add("phi_uncertain")
    ctx.trace.event("phi_done", n_pages=len(pages), n_detections=len(detections),
                    version=PHI_VERSION, duration_ms=round((time.perf_counter() - t0) * 1000.0, 3))
    return {"phi": {"masked_artifacts": artifacts, "detections": detections},
            "phi_vault": vault, "phi_masked_pages": payloads, "ocr_words": ocr_words,
            "pages": out_pages, "flags": sorted(flags)}

