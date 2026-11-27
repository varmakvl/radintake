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
from radintake.config import config_dir, env_flag
from radintake.core.types import BBox, Page
from radintake.orchestrator.stages import DOCUMENT_TYPES, StageContext, register_stage
from radintake.tools.model_client import MaskedPayload, ModelCallForbidden

PHI_VERSION = "phi-docs-1"
HEADER_TYPES = ("PATIENT_NAME", "PATIENT_ID", "DATE_OF_BIRTH", "ACCESSION_NUMBER")
_NAME_ANCHORS = ("PATIENT_NAME", "REFERRING_PHYSICIAN")

# Overlapping spans keep the type of the most specific layer.
_PRIORITY = {"anchor": 3, "known": 2, "regex": 1, "ner": 1, "handwriting": 1}
_TYPE_RANK = {t: 10 for t in HEADER_TYPES} | {"PERSON": 5, "DATE": 4, "ID": 3}

_BUILTIN: dict[str, Any] = {
    "anchors": [], "regex": [], "ner": "fallback",
    "honorifics": ["dr", "doctor", "prof", "professor", "mr", "mrs", "ms", "miss", "mx"],
    "mask": {"pad_px": 6, "fill": "background"},
    "handwriting": {"enabled": False},
}

# handwriting pass defaults (configs/phi.yaml `handwriting:` overrides them key by key)
_HW_DEFAULTS: dict[str, Any] = {
    "enabled": "auto", "model": "hw_reader",
    "credentials": r"^(?:RT|R\.T\.|RT\(R\)|RRT|ARRT|MD|M\.D\.|DO|RN|NP|PA-C)[,.;:]?$",
    "labels": r"\b(?:technologist|tech|signature|sign-?off)\s*:|\b(?:signed|read|performed|dictated)\s+by\b",
    "separators": r"^(?:[—–~=_|:\-]+|by)$",
    "max_name_words": 4, "low_conf": 0.6, "zone_frac": 0.3, "max_crops_per_page": 8,
    "crop_pad": 0.25, "word_overlap": 0.4, "max_new_tokens": 32,
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


def handwriting_params(params: dict[str, Any]) -> dict[str, Any]:
    hw = params.get("handwriting")
    return {**_HW_DEFAULTS, **(hw if isinstance(hw, dict) else {})}


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
        # Credential-anchored: OCR of a (handwritten) sign-off rarely yields a clean "Initial. Last"
        # ("— L. Broo ks, RT"), so whatever sits between a dash / "by" and an upper-case credential
        # on the same line is taken as a name, whatever its shape.
        self._cred_anchor = re.compile(
            r"(?:[—–~]|(?<![A-Za-z])-|\bby\b)\s*(?P<name>[^—–~\n]{2,40}?)\s*[,.]?\s*"
            r"(?:MD|M\.D\.|DO|RN|RT|R\.T\.|RRT|ARRT)(?![A-Za-z])")
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
                if t in _NAME_ANCHORS:
                    # a name value ends where an ID / accession / date starts: when OCR garbles the next
                    # label ("MRN:"), the name would otherwise swallow the next value as well
                    hits = [m.start() for _t, rx in self.regex for m in rx.finditer(line, end, stop)
                            if m.start() > end]
                    if hits:
                        stop = min(hits)
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
        for rx in (self._honorific, self._credential, self._cred_anchor):
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


# --------------------------------------------------------------------------- handwriting pass
def _conf(w: Any) -> float:
    return float(getattr(w, "conf", getattr(w, "confidence", 1.0)))


@dataclass
class HwCrop:
    """A region the handwriting reader reads: one or two line segments (a wrapped sign-off).

    Segments are stitched side by side into ONE line image and read once, so a name that wrapped
    ("S. Shep" / "herd, RT") is read as one line.
    """

    reason: str  # "anchor" (next to a credential / sign-off label) | "low_conf" (sign-off zone line)
    segments: list[tuple[int, list[int]]]  # (line index into the LineHits, word indexes left -> right)
    prefix: str = ""  # page context put before the read text: "— " after a dash, the label text

    def word_ids(self) -> list[int]:
        return [i for _, ids in self.segments for i in ids]


def select_handwriting_crops(words: Sequence[Any], hits: Sequence[LineHit], hp: dict[str, Any],
                             page_h: float) -> list[HwCrop]:
    """Crops for the handwriting reader, chosen AFTER the masking rules ran (``hits[*].spans``).

    Only two sources: (1) anchors - a credential token (``RT``, ``MD`` ...): the unmasked words
    between it and the previous separator (dash, ``=``, ``by``) or sentence end, at most
    ``max_name_words``, continuing onto the previous line when the name wrapped (the credential is
    kept in the crop so the detector's credential rules can fire); a sign-off label
    (``Technologist:``, ``Signed by`` ...) on a line without a credential: the unmasked words to its
    right, unless a rule already masked the first of them. (2) Low-confidence lines in the sign-off
    zone (from the first anchor line down, else the bottom ``zone_frac`` of the page): the run of
    words below ``low_conf`` plus one neighbour each side. Words already masked by a rule are never
    part of a crop, and a word is in at most one crop.
    """
    cred = re.compile(hp["credentials"])
    label = re.compile(hp["labels"], re.IGNORECASE)
    sep = re.compile(hp["separators"], re.IGNORECASE)
    max_w = int(hp["max_name_words"])
    masked = {i for h in hits for s in h.spans for i in h.words_of(s)}
    order = [[i for _, _, i in h.offsets] for h in hits]

    def stop(i: int) -> bool:  # left boundary of a name: separator, masked word, sentence end
        t = words[i].text.strip()
        return (i in masked or bool(sep.match(t)) or bool(cred.match(t))
                or (len(t) > 3 and t.endswith((".", ";", ":")) and t[:1].islower()))

    crops: list[HwCrop] = []
    used: set[int] = set()
    first_signoff: int | None = None
    for li, ids in enumerate(order):
        for k, i in enumerate(ids):
            if not cred.match(words[i].text.strip()):
                continue
            if first_signoff is None:
                first_signoff = li
            segs: list[tuple[int, list[int]]] = []
            prefix, n = "", 0
            cur_line, j, cur = li, k - 1, [i]
            while n < max_w:
                if j < 0:  # reached the line start: the name may have wrapped from the previous line
                    if cur_line != li or cur_line == 0 or not order[cur_line - 1]:
                        break
                    segs.insert(0, (cur_line, cur))
                    cur_line, cur = cur_line - 1, []
                    j = len(order[cur_line]) - 1
                    continue
                wi = order[cur_line][j]
                if stop(wi):
                    prefix = "— " if sep.match(words[wi].text.strip()) else ""
                    break
                cur.insert(0, wi)
                n += 1
                j -= 1
            if cur:
                segs.insert(0, (cur_line, cur))
            name_ids = [w for _, s in segs for w in s if w != i]
            if not name_ids or any(w in used for w in name_ids):
                continue
            if re.match(r"^[—–~=_\-]", words[name_ids[0]].text.strip()):  # dash glued to the name ("__S.")
                prefix = "— "
            crops.append(HwCrop("anchor", segs, prefix=prefix))
            used.update(name_ids)
        if first_signoff is None and label.search(hits[li].text):
            first_signoff = li
    for li, ids in enumerate(order):  # sign-off labels on lines without a credential
        m = label.search(hits[li].text)
        if not m or any(cred.match(words[i].text.strip()) for i in ids):
            continue
        cur = []
        for i in [i for a, _, i in hits[li].offsets if a >= m.end()]:
            t = words[i].text.strip()
            if not cur and (sep.match(t) or t == ":"):
                continue
            if i in masked or i in used:  # the name ends here; right after the label: a rule found it
                if len(cur) <= 1:  # "Signed by Dr. [masked]": nothing left to read
                    cur = []
                break
            cur.append(i)
            if len(cur) >= max_w:
                break
        if cur:
            crops.append(HwCrop("anchor", [(li, cur)], prefix=m.group(0) + " "))
            used.update(cur)
    if first_signoff is None:
        zone = [li for li, ids in enumerate(order)
                if ids and words[ids[0]].bbox.y0 >= (1.0 - float(hp["zone_frac"])) * page_h]
    else:
        zone = list(range(first_signoff, len(order)))
    low = float(hp["low_conf"])
    for li in zone:  # low-confidence lines inside the sign-off zone
        ids = order[li]
        lowk = [k for k, i in enumerate(ids) if _conf(words[i]) < low and i not in masked and i not in used]
        if not lowk:
            continue
        a, b = max(0, lowk[0] - 1), min(len(ids), lowk[-1] + 2)
        seg = [i for i in ids[a:b] if i not in masked and i not in used][: max_w + 2]
        if seg:
            crops.append(HwCrop("low_conf", [(li, seg)]))
            used.update(seg)
    return crops[: int(hp["max_crops_per_page"])]


def _crop(img: np.ndarray, box: BBox, pad_frac: float) -> np.ndarray:
    h, w = img.shape[:2]
    p = pad_frac * box.height
    x0, y0 = int(max(0, np.floor(box.x0 - p))), int(max(0, np.floor(box.y0 - p)))
    x1, y1 = int(min(w, np.ceil(box.x1 + p))), int(min(h, np.ceil(box.y1 + p)))
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def crop_line_image(image: np.ndarray, crop: HwCrop, words: Sequence[Any],
                    pad_frac: float) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """The crop's segments cut from ``image`` and stitched left -> right into one line image.

    Returns the image and each segment's ``(start, end)`` as fractions of its width.
    """
    parts = [_crop(image, _union([words[i].bbox for i in ids]), pad_frac) for _, ids in crop.segments]
    if len(parts) == 1:
        return parts[0], [(0.0, 1.0)]
    hgt = max(p.shape[0] for p in parts)
    gap = max(4, hgt // 3)
    fill = _fill_value(image, "background")
    tiles: list[np.ndarray] = []
    for k, p in enumerate(parts):
        t = np.empty((hgt, p.shape[1]) + p.shape[2:], p.dtype)
        t[...] = fill
        top = (hgt - p.shape[0]) // 2
        t[top:top + p.shape[0]] = p
        if k:
            g = np.empty((hgt, gap) + p.shape[2:], p.dtype)
            g[...] = fill
            tiles.append(g)
        tiles.append(t)
    out = np.ascontiguousarray(np.concatenate(tiles, axis=1))
    ext, x = [], 0
    for k, p in enumerate(parts):
        x += gap if k else 0
        ext.append((x / out.shape[1], (x + p.shape[1]) / out.shape[1]))
        x += p.shape[1]
    return out, ext


_NAME_TYPES = ("PERSON", "PATIENT_NAME", "REFERRING_PHYSICIAN")


def handwriting_spans(crop: HwCrop, text: str, extents: Sequence[tuple[float, float]], words: Sequence[Any],
                      hits: Sequence[LineHit], detector: PhiDetector, hp: dict[str, Any]) -> list[tuple[int, Span]]:
    """Run the detector on the recognised crop text; map name spans back to OCR words.

    ``crop.prefix + text`` goes through the detector. A name span's character range (as a fraction
    of ``text``) is mapped onto the stitched line image, then onto each segment's word box via
    ``extents``, and snapped to the segment's name words (credential and separator tokens excluded)
    that it covers by at least ``word_overlap`` of their width. Returns ``(line index, Span over that
    line's text)`` pairs with ``source="handwriting"``.
    """
    text = text.strip()
    if not text:
        return []
    cred = re.compile(hp["credentials"])
    sep = re.compile(hp["separators"], re.IGNORECASE)
    pre = crop.prefix
    spans = [s for s in detector.detect(pre + text) if s.type in _NAME_TYPES and s.end > len(pre)]
    picked: dict[int, set[int]] = {}
    for s in spans:
        fa = max(0, s.start - len(pre)) / len(text)
        fb = min(len(text), s.end - len(pre)) / len(text)
        for (li, ids), (g0, g1) in zip(crop.segments, extents):
            a, b = max(fa, g0), min(fb, g1)
            if b <= a:
                continue
            box = _union([words[i].bbox for i in ids])
            xa = box.x0 + (a - g0) / (g1 - g0) * box.width
            xb = box.x0 + (b - g0) / (g1 - g0) * box.width
            cand = [i for i in ids if not cred.match(words[i].text.strip()) and not sep.match(words[i].text.strip())]
            ov = {i: max(0.0, min(xb, words[i].bbox.x1) - max(xa, words[i].bbox.x0)) / max(words[i].bbox.width, 1e-6)
                  for i in cand}
            sel = {i for i, f in ov.items() if f >= float(hp["word_overlap"])}
            if not sel and ov and max(ov.values()) > 0:
                sel = {max(ov, key=ov.get)}
            if sel:
                picked.setdefault(li, set()).update(sel)
    out: list[tuple[int, Span]] = []
    for li, sel in sorted(picked.items()):
        run: list[tuple[int, int]] = []  # one span per contiguous run of picked words on the line
        for a, b, i in hits[li].offsets:
            if i in sel:
                run.append((a, b))
            elif run:
                out.append((li, Span(run[0][0], run[-1][1], "PERSON", "handwriting")))
                run = []
        if run:
            out.append((li, Span(run[0][0], run[-1][1], "PERSON", "handwriting")))
    return out


class HandwritingPass:
    """Per-case state of the handwriting pass: the reader (or why it is skipped) and counters."""

    def __init__(self, reader: Any, hp: dict[str, Any], *, skip_reason: str | None = None, trace: Any = None):
        self.reader = reader
        self.hp = hp
        self.skip_reason = skip_reason or (None if reader is not None else "disabled")
        self.trace = trace
        self.stats = {"n_crops": 0, "n_anchor_crops": 0, "n_low_conf_crops": 0, "n_reads": 0,
                      "n_detections": 0, "read_ms": 0.0}

    @property
    def active(self) -> bool:
        return self.reader is not None and self.skip_reason is None

    def run(self, page: Page, words: Sequence[Any], hits: list[LineHit], detector: PhiDetector,
            image: np.ndarray) -> list[tuple[HwCrop, list[tuple[int, Span]]]]:
        """Read the crops of one page; ``image`` is the processed page with rule detections FILLED."""
        from radintake.tools.handwriting import HandwritingUnavailable, PhiCrop

        if not self.active:
            return []
        crops = select_handwriting_crops(words, hits, self.hp, float(page.image.shape[0]))
        out: list[tuple[HwCrop, list[tuple[int, Span]]]] = []
        for c in crops:
            self.stats["n_crops"] += 1
            self.stats["n_anchor_crops" if c.reason == "anchor" else "n_low_conf_crops"] += 1
            t0 = time.perf_counter()
            try:
                img, extents = crop_line_image(image, c, words, float(self.hp["crop_pad"]))
                payload = PhiCrop(img, phi_crop=True, meta={"page": page.index, "reason": c.reason})
                text = self.reader.read(payload, trace=self.trace)
                self.stats["n_reads"] += 1
            except HandwritingUnavailable as exc:
                self.skip_reason = exc.reason
                return out
            except ModelCallForbidden:
                self.skip_reason = "model_calls_forbidden"
                return out
            except Exception as exc:  # noqa: BLE001 - a failing reader never fails the PHI stage
                self.skip_reason = f"error:{type(exc).__name__}"
                return out
            finally:
                self.stats["read_ms"] += (time.perf_counter() - t0) * 1000.0
            if text.strip():  # recognised text is PHI: scrub it from any log / trace
                from radintake.logging import register_phi

                register_phi(text.strip())
            spans = handwriting_spans(c, text, extents, words, hits, detector, self.hp)
            if spans:
                out.append((c, spans))
        return out


def make_handwriting_pass(params: dict[str, Any], config: Any = None, trace: Any = None) -> HandwritingPass:
    """Build the pass from ``phi.handwriting``; never raises (an unusable reader -> skip reason)."""
    hp = handwriting_params(params)
    enabled = hp.get("enabled", "auto")
    if enabled is False or str(enabled).lower() in ("false", "off", "0", "no"):
        return HandwritingPass(None, hp, skip_reason="disabled", trace=trace)
    try:
        from radintake.tools.handwriting import HandwritingReader

        reader = HandwritingReader(str(hp["model"]), config=config, trace=trace,
                                   max_new_tokens=int(hp["max_new_tokens"]))
    except Exception as exc:  # noqa: BLE001 - unknown model / bad registry -> skip
        return HandwritingPass(None, hp, skip_reason=f"reader_error:{type(exc).__name__}", trace=trace)
    reason = reader.unavailable_reason()
    if reason is None and env_flag("RADINTAKE_FORBID_MODEL_CALLS", bool(getattr(config, "forbid_model_calls", False))):
        reason = "model_calls_forbidden"
    return HandwritingPass(None if reason else reader, hp, skip_reason=reason, trace=trace)


def mask_page(page: Page, original: Page | None, words: Sequence[Any], detector: PhiDetector,
              vault: PhiVault, *, pad: float = 6, fill: Any = "background",
              known_tokens: Iterable[str] = (), handwriting: HandwritingPass | None = None) -> PageMaskResult:
    """Detect PHI on ``words`` (processed-page coords) and mask both page images and the text.

    The masking rules run first; when ``handwriting`` is active its reader then sees only crops
    of the processed page with every rule detection already filled, and adds name detections.
    """
    original = original or page
    dets: list[Detection] = []
    masked_words: set[int] = set()
    lines_out: list[str] = []
    hits = detect_words(words, detector, known_tokens)
    hw_groups: list[list[tuple[int, Span]]] = []
    if handwriting is not None and handwriting.active:
        rule_boxes = [words[i].bbox for h in hits for s in h.spans for i in h.words_of(s)]
        rule_masked = mask_image(page.image, rule_boxes, pad=pad, fill=fill)
        for _crop_, spans in handwriting.run(page, words, hits, detector, rule_masked):
            for li, s in spans:
                hits[li].spans = _merge_spans(hits[li].spans + [s])
            hw_groups.append(spans)
    hw_dets: dict[tuple[int, int, int], int] = {}  # (line, start, end) of a handwriting span -> group
    for g, spans in enumerate(hw_groups):
        for li, s in spans:
            hw_dets[(li, s.start, s.end)] = g
    group_words: dict[int, list[int]] = {}
    group_ph: dict[int, str] = {}
    for li, h in enumerate(hits):
        parts, pos = [], 0
        for s in h.spans:
            idx = h.words_of(s)
            ph = vault.placeholder(s.type, h.text[s.start:s.end])
            g = hw_dets.get((li, s.start, s.end)) if s.source == "handwriting" else None
            if g is not None:  # one detection per handwriting crop (a wrapped name spans two lines)
                group_words.setdefault(g, []).extend(idx)
                group_ph.setdefault(g, ph)
            else:
                cur = _union([words[i].bbox for i in idx])
                conf = float(min(_conf(words[i]) for i in idx))
                dets.append(Detection(s.type, page.index, page.to_original(cur), cur, s.source, ph, conf))
            masked_words.update(idx)
            parts += [h.text[pos:s.start], ph]
            pos = s.end
        parts.append(h.text[pos:])
        lines_out.append("".join(parts))
    for g, idx in sorted(group_words.items()):
        cur = _union([words[i].bbox for i in idx])
        dets.append(Detection("PERSON", page.index, page.to_original(cur), cur, "handwriting", group_ph[g],
                              float(min(_conf(words[i]) for i in idx))))
    if handwriting is not None:
        handwriting.stats["n_detections"] += len(group_words)
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
    from radintake.tools.pdf_text import page_words

    params = load_params(ctx.config)
    mcfg = params.get("mask") or {}
    pad, fill = float(mcfg.get("pad_px", 6)), mcfg.get("fill", "background")
    detector = PhiDetector(params)
    vault = PhiVault()
    hw = make_handwriting_pass(params, ctx.config, ctx.trace)
    if not hw.active:
        ctx.trace.event("phi_handwriting", status="skipped", reason=hw.skip_reason,
                        model_id=str(hw.hp.get("model")))
    pages: list[Page] = list(ctx.data.get("pages") or [])
    originals = {p.index: p for p in (ctx.data.get("original_pages") or pages)}
    rel_dir = f"{ctx.input_name}.masked"
    t0 = time.perf_counter()

    artifacts: list[str] = []
    detections: list[dict[str, Any]] = []
    payloads: list[MaskedPayload] = []
    ocr_words: dict[int, list[Any]] = {}
    cached: dict[int, list[Any]] = ctx.data.get("ocr_words") or {}
    out_pages: list[Page] = []
    flags: set[str] = set()
    known: set[str] = set()
    for pg in pages:
        words = cached.get(pg.index)  # from the tables stage (text layer or OCR + re-read cells)
        if words is None:
            words = page_words(pg, trace=ctx.trace)
        ocr_words[pg.index] = words
        if not words:
            flags.add("phi_uncertain")  # nothing readable: cannot vouch for this page
        was_active = hw.active
        before = dict(hw.stats)
        res = mask_page(pg, originals.get(pg.index), words, detector, vault, pad=pad, fill=fill,
                        known_tokens=known, handwriting=hw)
        if was_active:  # counts, timings and the model id only - never recognised text
            delta = {k: (round(v - before[k], 3) if isinstance(v, float) else v - before[k])
                     for k, v in hw.stats.items()}
            ctx.trace.event("phi_handwriting", page=pg.index, status="ok" if hw.active else "skipped",
                            reason=hw.skip_reason, model_id=getattr(hw.reader, "model_id", None), **delta)
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

