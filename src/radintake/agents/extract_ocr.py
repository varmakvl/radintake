"""OCR-first extraction: OCR words -> lines -> label-anchored field mapping -> report.v1.

Pipeline (PLAN.md 2.4, OCR-first)::

    OCR words of each preprocessed page (reused from the PHI stage, else OCR'd here)
      -> lines in reading order (Tesseract line ids, else y-overlap)
      -> header labels   (Patient Name / Patient ID / Date of Birth / Accession No.; the same
                          anchor patterns as the PHI agent, configs/phi.yaml) -> value to the right
      -> exam table      (a header row with "Exam Date" / "Modality" cells; values by column)
      -> section headings (INDICATION / COMPARISON / FINDINGS / IMPRESSION and aliases, fuzzy)
                          -> following lines until the next heading, a stop line, or a labelled
                          non-continuation line at the top of a later page (sections may span
                          a page break: provenance then has one entry per page)

Every field is ``{value, confidence, provenance}``:

* ``confidence`` = mean OCR word confidence of the value words x mapping confidence (1.0 for an
  exact heading, the fuzzy ratio for an OCR-damaged one, fixed values for anchors and tables);
* ``provenance`` = one entry per page: the union of the value word boxes, mapped from the
  processed page back to ORIGINAL page coordinates through the preprocess transform
  (:func:`radintake.agents.preprocess.map_to_original`), ``source: "ocr"``.

Header values are read locally from the UNMASKED page words and never leave the process (the
trace only gets hashes). Free text is kept as data: an instruction-like sentence inside a section
("Ignore previous instructions ...") is never obeyed and never dropped - it is moved verbatim,
with its own provenance, into the field's ``untrusted_text`` list so the value holds the clinical
text and the verifier / injection detector can flag the page.

Handwritten / low-confidence regions: a crop re-reader (TrOCR) is optional and only used when a
``reread`` callable is supplied; none is bundled (no offline TrOCR weights that fit the budget are
shipped), so such fields are marked ``low_confidence`` and the case gets the ``low_confidence``
flag. This module makes no model or network call.
"""

from __future__ import annotations

import re
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

from radintake.agents import preprocess as _preprocess
from radintake.core.types import BBox, Page
from radintake.models.report import (
    ALL_FIELDS,
    HEADER_FIELDS,
    SECTION_FIELDS,
    Provenance,
    Report,
    ReportField,
    UntrustedText,
)

EXTRACT_OCR_VERSION = "extract-ocr-1"

DEFAULT_HEADINGS: dict[str, list[str]] = {
    "indication": ["INDICATION", "INDICATIONS", "CLINICAL INDICATION", "CLINICAL HISTORY", "HISTORY",
                   "REASON FOR EXAM", "REASON FOR STUDY"],
    "comparison": ["COMPARISON", "COMPARISONS", "PRIOR STUDIES", "PRIORS"],
    "findings": ["FINDINGS", "FINDING"],
    "impression": ["IMPRESSION", "IMPRESSIONS", "CONCLUSION", "CONCLUSIONS"],
}
DEFAULT_TABLE_LABELS: dict[str, list[str]] = {
    "exam_date": ["EXAM DATE", "DATE OF EXAM", "STUDY DATE", "EXAMINATION DATE"],
    "modality": ["MODALITY"],
}
# Extra column labels used only for DETECTED tables (tools.tables): in the line-based fallback an
# "Accession No.: ..." header line would itself look like a table header row.
DETECTED_TABLE_LABELS: dict[str, list[str]] = {
    "accession_number": ["ACCESSION", "ACCESSION NO", "ACCESSION NUMBER", "ACC NO"],
}
# Instruction-like sentences inside report text: kept as data, never part of the clinical value.
DEFAULT_UNTRUSTED_PATTERNS: list[str] = [
    (r"\b(?:ignore|disregard|forget|override)\b.{0,40}\b(?:previous|prior|above|earlier|all|any)\b"
     r".{0,30}\b(?:instructions?|prompts?|rules?|directions?)\b"),
    r"^\s*(?:system|assistant|user|developer)\s*:",
    r"\b(?:you are now|act as|pretend to be|roleplay as)\b",
    r"\b(?:output|print|reveal|return|list|show)\b.{0,30}\bpatient(?:'s)?\s+(?:name|id|identity|details)\b",
    r"\bmark (?:this|the) (?:study|report|exam|examination|case)\b.{0,30}\bas\b",
]
DEFAULT_STOP_PATTERNS: list[str] = [
    r"^(?:electronically\s+)?signed\b", r"^dictated\b", r"^transcribed\b", r"^reported\s+by\b",
    r"^(?:attending\s+)?radiologist\s*:", r"^end\s+of\s+report\b",
    # technologist / tech notes and addenda are not part of the clinical section above them
    r"^tech(?:nologist|nician)?\.?\s*(?:note|notes|comments?|remarks?)\b", r"^addendum\b",
]
# Exam-details fallback when the table header row is unreadable (e.g. shaded header cells):
# the header block's dates that are not the date of birth, and known modality codes.
_DATE_TOKEN = re.compile(r"^(?:\d{1,2}[/.-]\d{1,2}[/.-](?:19|20)?\d{2}|(?:19|20)\d{2}[/.-]\d{1,2}[/.-]\d{1,2})$")
_MODALITIES = {"CR", "DX", "CT", "MR", "MRI", "US", "NM", "PT", "MG", "XR", "RF", "OT"}
# a token that is itself a "Label:" (e.g. "Referring:") ends the value of the label before it
_LABEL_TOKEN = re.compile(r"^[A-Za-z][A-Za-z.]{1,24}:$")
_PAGE_FURNITURE = re.compile(r"^\s*page\s+\d+(?:\s*(?:of|/)\s*\d+)?\s*[-–—:|.]?\s*", re.IGNORECASE)
_LABEL = re.compile(r"^\s*([A-Za-z][A-Za-z/&'() ]{1,40}?)\s*:")
_BORDER = re.compile(r"^[|¦_~=]+$")
_ALNUM = re.compile(r"[A-Za-z0-9]")

DEFAULTS: dict[str, Any] = {
    "heading_min_ratio": 85,          # fuzzy match for OCR-damaged headings
    "header_mapping_conf": 0.98,
    "table_mapping_conf": 0.9,
    "low_conf_threshold": 0.5,        # mean word conf below this -> low_confidence
    "max_heading_words": 4,
    "headings": DEFAULT_HEADINGS,
    "table_labels": DEFAULT_TABLE_LABELS,
    "untrusted_patterns": DEFAULT_UNTRUSTED_PATTERNS,
    "stop_patterns": DEFAULT_STOP_PATTERNS,
}

_ANCHOR_FIELD = {"PATIENT_NAME": "patient_name", "PATIENT_ID": "patient_id",
                 "DATE_OF_BIRTH": "date_of_birth", "ACCESSION_NUMBER": "accession_number"}


def load_params(cfg: Any = None) -> dict[str, Any]:
    """Defaults < the ``extraction.ocr_first`` section of the run config."""
    raw = getattr(cfg, "raw", None) or {}
    ext = raw.get("extraction") if isinstance(raw.get("extraction"), dict) else {}
    user = ext.get("ocr_first") if isinstance(ext.get("ocr_first"), dict) else {}
    return {**DEFAULTS, **user}


# --------------------------------------------------------------------------- tokens / lines
@dataclass
class Token:
    text: str
    bbox: BBox  # processed-page coordinates
    conf: float
    page: int
    eol: bool = False  # last token of its OCR line


@dataclass
class Line:
    page: int
    tokens: list[Token]

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def bbox(self) -> BBox:
        return union([t.bbox for t in self.tokens])

    @property
    def height(self) -> float:
        return statistics.median(t.bbox.height for t in self.tokens)


def union(boxes: Sequence[BBox]) -> BBox:
    return BBox(min(b.x0 for b in boxes), min(b.y0 for b in boxes),
                max(b.x1 for b in boxes), max(b.y1 for b in boxes))


def _conf(w: Any) -> float:
    c = getattr(w, "conf", None)
    if c is None:
        c = getattr(w, "confidence", 1.0)
    return float(min(max(c, 0.0), 1.0))


def page_lines(page_index: int, words: Sequence[Any]) -> list[Line]:
    """Group words into lines (Tesseract line ids when present, else y overlap), reading order."""
    words = [w for w in words if (w.text or "").strip() and not _BORDER.match(w.text.strip())]
    groups: list[list[Any]] = []
    if words and hasattr(words[0], "line_id"):
        by_id: dict[Any, list[Any]] = {}
        for w in words:
            by_id.setdefault(w.line_id, []).append(w)
        groups = list(by_id.values())
    else:
        for w in sorted(words, key=lambda w: (w.bbox.y0, w.bbox.x0)):
            for g in groups:
                r = g[-1].bbox
                ov = min(w.bbox.y1, r.y1) - max(w.bbox.y0, r.y0)
                if ov > 0.5 * min(w.bbox.height, r.height):
                    g.append(w)
                    break
            else:
                groups.append([w])
    lines: list[Line] = []
    for g in groups:
        g = sorted(g, key=lambda w: w.bbox.x0)
        toks = [Token(w.text.strip(), w.bbox, _conf(w), page_index) for w in g]
        toks[-1].eol = True
        lines.append(Line(page_index, toks))
    lines.sort(key=lambda ln: (ln.bbox.y0, ln.bbox.x0))
    return lines


def join_tokens(tokens: Sequence[Token]) -> str:
    """Join tokens with spaces; a line-final hyphen before a lowercase word is de-hyphenated."""
    out = ""
    for i, t in enumerate(tokens):
        if i == 0:
            out = t.text
            continue
        prev = tokens[i - 1]
        if prev.eol and prev.text.endswith("-") and len(prev.text) > 1 and t.text[:1].islower():
            out = out[:-1] + t.text
        else:
            out += " " + t.text
    return out


def _norm_label(text: str) -> str:
    return re.sub(r"[^A-Z ]+", " ", text.upper()).strip()


# --------------------------------------------------------------------------- mapping
@dataclass
class FieldHit:
    tokens: list[Token]
    mapping_conf: float
    untrusted: list[tuple[list[Token], str]] = field(default_factory=list)


class OcrFirstExtractor:
    def __init__(self, params: dict[str, Any] | None = None, anchors: Sequence[tuple[str, Any]] | None = None):
        self.p = {**DEFAULTS, **(params or {})}
        if anchors is None:
            from radintake.agents.phi import load_params as phi_params

            anchors = [(a["type"].upper(), re.compile(a["pattern"], re.IGNORECASE))
                       for a in phi_params().get("anchors") or []]
        self.anchors = [(_ANCHOR_FIELD[t], rx) for t, rx in anchors if t in _ANCHOR_FIELD]
        self.headings = {k: [_norm_label(a) for a in v] for k, v in self.p["headings"].items()}
        self.table_labels = {k: [_norm_label(a) for a in v] for k, v in self.p["table_labels"].items()}
        self.untrusted = [re.compile(x, re.IGNORECASE) for x in self.p["untrusted_patterns"]]
        self.stops = [re.compile(x, re.IGNORECASE) for x in self.p["stop_patterns"]]

    # -- headings ---------------------------------------------------------------------
    def heading_of(self, line: Line) -> tuple[str, float, int] | None:
        """(section, mapping conf, n heading tokens) when ``line`` starts with a section heading."""
        toks = line.tokens
        n_max = min(self.p["max_heading_words"], len(toks))
        colon_at = next((i for i in range(n_max) if toks[i].text.endswith(":")), None)
        if colon_at is not None:
            cands = [colon_at + 1]
        else:  # no colon: the whole (short) line must be an upper-case heading
            if len(toks) > n_max or not line.text.isupper():
                return None
            cands = [len(toks)]
        best: tuple[str, float, int] | None = None
        for n in cands:
            label = _norm_label(" ".join(t.text for t in toks[:n]))
            if not label:
                continue
            for sec, aliases in self.headings.items():
                r = max(fuzz.ratio(label, a) for a in aliases)
                if r >= self.p["heading_min_ratio"] and (best is None or r / 100.0 > best[1]):
                    best = (sec, r / 100.0, n)
        return best

    def header_hits(self, line: Line) -> list[tuple[str, list[Token]]]:
        """Header anchors on ``line``: (field, value tokens right of the label up to the next one)."""
        text, offs, pos = "", [], 0
        for t in line.tokens:
            offs.append((pos, pos + len(t.text)))
            text += t.text + " "
            pos += len(t.text) + 1
        labels = sorted(((m.start(), m.end(), f) for f, rx in self.anchors for m in rx.finditer(text)),
                        key=lambda x: (x[0], -x[1]))
        kept: list[tuple[int, int, str]] = []
        for lab in labels:
            if kept and lab[0] < kept[-1][1]:
                continue
            kept.append(lab)
        out: list[tuple[str, list[Token]]] = []
        for i, (_, end, f) in enumerate(kept):
            stop = kept[i + 1][0] if i + 1 < len(kept) else len(text)
            vals: list[Token] = []
            h = line.height
            for (a, b), t in zip(offs, line.tokens, strict=True):
                if b <= end or a >= stop:
                    continue
                txt = t.text[max(0, end - a):] if a < end else t.text
                txt = txt.lstrip(" \t:#.")
                if not txt:
                    continue
                # the value ends at an unknown "Label:" token or a column gap (two-column headers,
                # e.g. "Patient Name: Doe, Jane      Referring: Dr. X")
                if vals and (_LABEL_TOKEN.match(txt) or t.bbox.x0 - vals[-1].bbox.x1 > 2.5 * h):
                    break
                vals.append(Token(txt, t.bbox, t.conf, t.page, t.eol))
            if vals:
                out.append((f, vals))
        return out

    def exam_fallback(self, lines: list[Line], consumed: set[int], hits: dict[str, FieldHit]) -> None:
        """Exam date / modality from the header block when no readable table header gave them."""
        first_head = next((i for i, ln in enumerate(lines) if self.heading_of(ln)), len(lines))
        dob = {t.text for t in (hits.get("date_of_birth").tokens if hits.get("date_of_birth") else [])}
        page1 = lines[0].page if lines else 1
        for i, ln in enumerate(lines[:first_head]):
            if i in consumed or ln.page != page1:
                continue
            for t in ln.tokens:
                txt = t.text.strip(".,;:()")
                if "exam_date" not in hits and _DATE_TOKEN.match(txt) and txt not in dob:
                    hits["exam_date"] = FieldHit([Token(txt, t.bbox, t.conf, t.page, t.eol)], 0.75)
                if "modality" not in hits and txt.upper() in _MODALITIES and txt.isupper():
                    hits["modality"] = FieldHit([Token(txt, t.bbox, t.conf, t.page, t.eol)], 0.75)

    def is_stop(self, text: str) -> bool:
        return any(rx.search(text) for rx in self.stops)

    # -- table ------------------------------------------------------------------------
    def table_hits(self, lines: list[Line], idx: int) -> tuple[dict[str, list[Token]], int] | None:
        """If ``lines[idx]`` is an exam-table header row, map the next row's tokens by column.

        Returns (field -> value tokens, index of the value row)."""
        line = lines[idx]
        h = line.height
        cells: list[list[Token]] = []
        for t in line.tokens:
            if cells and t.bbox.x0 - cells[-1][-1].bbox.x1 <= 1.5 * h:
                cells[-1].append(t)
            else:
                cells.append([t])
        if len(cells) < 2:
            return None
        labels: list[str | None] = []
        for c in cells:
            lab = _norm_label(" ".join(t.text for t in c))
            best, br = None, 0.0
            for f, aliases in self.table_labels.items():
                r = max(fuzz.ratio(lab, a) for a in aliases)
                if r >= self.p["heading_min_ratio"] and r > br:
                    best, br = f, r
            labels.append(best)
        if not any(labels):
            return None
        j = idx + 1
        if j >= len(lines) or lines[j].page != line.page or lines[j].bbox.y0 - line.bbox.y1 > 4 * h:
            return None
        nxt = lines[j]
        starts = [c[0].bbox.x0 - 0.5 * h for c in cells]
        out: dict[str, list[Token]] = {}
        for t in nxt.tokens:
            col = max((i for i, s in enumerate(starts) if s <= t.bbox.x0), default=0)
            if labels[col]:
                out.setdefault(labels[col], []).append(t)
        return out, j

    # -- untrusted text ---------------------------------------------------------------
    def split_untrusted(self, tokens: list[Token]) -> tuple[list[Token], list[tuple[list[Token], str]]]:
        sentences: list[list[Token]] = [[]]
        for t in tokens:
            sentences[-1].append(t)
            if re.search(r"[.!?][\"')\]]*$", t.text):
                sentences.append([])
        kept: list[Token] = []
        untrusted: list[tuple[list[Token], str]] = []
        for s in sentences:
            if not s:
                continue
            txt = join_tokens(s)
            if any(rx.search(txt) for rx in self.untrusted):
                untrusted.append((s, "instruction_like_text"))
            else:
                kept += s
        return kept, untrusted

    # -- main -------------------------------------------------------------------------
    def table_field_hits(self, tables: dict[int, list[Any]]) -> dict[str, FieldHit]:
        """Header-labelled columns of detected tables (row 0 = labels, row 1 = values)."""
        hits: dict[str, FieldHit] = {}
        labels = {**{k: [_norm_label(a) for a in v] for k, v in DETECTED_TABLE_LABELS.items()}, **self.table_labels}
        for pidx in sorted(tables):
            for t in tables[pidx]:
                if t.n_rows < 2:
                    continue
                grid = {(c.row, c.col): c for c in t.cells}
                for col in range(t.n_cols):
                    head, val = grid.get((0, col)), grid.get((1, col))
                    if head is None or val is None or not val.text:
                        continue
                    lab = _norm_label(head.text)
                    best, br = None, 0.0
                    for f, aliases in labels.items():
                        r = max(fuzz.ratio(lab, a) for a in aliases)
                        if r >= self.p["heading_min_ratio"] and r > br:
                            best, br = f, r
                    if best and best not in hits:
                        hits[best] = FieldHit([Token(val.text, val.bbox, val.confidence, pidx, True)],
                                              self.p["table_mapping_conf"])
        return hits

    def map_fields(self, lines: list[Line], tables: dict[int, list[Any]] | None = None) -> dict[str, FieldHit]:
        tables = tables or {}
        table_hits = self.table_field_hits(tables)
        # exam date / modality: a detected table column wins (its cells were re-read); accession: a
        # labelled header line wins and a table column only fills it when no header line has it
        late = {f: table_hits.pop(f) for f in list(table_hits) if f in DETECTED_TABLE_LABELS}
        hits: dict[str, FieldHit] = dict(table_hits)
        consumed: set[int] = set()
        self.table_sections: dict[tuple[int, int], str] = {}

        def table_of(tok: Token) -> int | None:
            cx, cy = (tok.bbox.x0 + tok.bbox.x1) / 2, (tok.bbox.y0 + tok.bbox.y1) / 2
            for ti, t in enumerate(tables.get(tok.page) or []):
                b = t.bbox
                if b.x0 <= cx <= b.x1 and b.y0 <= cy <= b.y1:
                    return ti
            return None
        # exam table + header anchors (first occurrence in reading order wins)
        for i, ln in enumerate(lines):
            if i in consumed:
                continue
            tab = self.table_hits(lines, i)
            if tab:
                consumed.update((i, tab[1]))
                for f, toks in tab[0].items():
                    hits.setdefault(f, FieldHit(toks, self.p["table_mapping_conf"]))
                continue
            if self.heading_of(ln):
                continue
            hh = self.header_hits(ln)
            if hh:
                consumed.add(i)
                for f, toks in hh:
                    hits.setdefault(f, FieldHit(toks, self.p["header_mapping_conf"]))

        for f, h in late.items():
            hits.setdefault(f, h)
        self.exam_fallback(lines, consumed, hits)

        # letterhead lines above the first heading on page 1 are page furniture when repeated
        first_head = next((i for i, ln in enumerate(lines) if self.heading_of(ln)), len(lines))
        letterhead = [ln.text for i, ln in enumerate(lines[:first_head]) if i not in consumed]

        cur: str | None = None
        cur_page = 0
        cur_x0 = 0.0
        sections: dict[str, tuple[list[Token], float]] = {}
        for i, ln in enumerate(lines):
            if i in consumed:
                continue
            head = self.heading_of(ln)
            if head:
                sec, conf, n = head
                cur, cur_page, cur_x0 = sec, ln.page, ln.bbox.x0
                if sec not in sections:
                    sections[sec] = ([], conf)
                else:  # repeated heading (e.g. on a continuation page): keep appending
                    sections[sec] = (sections[sec][0], min(sections[sec][1], conf))
                sections[sec][0].extend(ln.tokens[n:])
                continue
            if cur is None:
                continue
            text = ln.text
            stripped = _PAGE_FURNITURE.sub("", text)
            if self.is_stop(stripped):
                cur = None
                continue
            if ln.page != cur_page:
                if not stripped or any(fuzz.ratio(text, lh) >= 90 for lh in letterhead):
                    continue  # page furniture / repeated letterhead
                if _LABEL.match(stripped):  # a labelled line at the top of a new page is not a continuation
                    cur = None
                    continue
                cur_page = ln.page
            elif _LABEL.match(text) and _LABEL.match(text).group(1).isupper() \
                    and ln.bbox.x0 <= cur_x0 + 0.5 * ln.height:
                cur = None  # an unknown upper-case section heading ends the current section
                continue
            toks = ln.tokens
            if stripped != text:  # drop the "Page N of M -" tokens
                n_drop = len(text.split()) - len(stripped.split())
                toks = toks[n_drop:]
            if tables:  # table cells are reported as tables, not flattened into the section's prose
                kept_toks = []
                for t in toks:
                    ti = table_of(t)
                    if ti is None:
                        kept_toks.append(t)
                    else:
                        self.table_sections.setdefault((t.page, ti), cur)
                toks = kept_toks
            sections[cur][0].extend(toks)

        for sec, (toks, conf) in sections.items():
            kept, untrusted = self.split_untrusted(toks)
            hits[sec] = FieldHit(kept, conf, untrusted)
        return hits


# --------------------------------------------------------------------------- report building
def provenance(tokens: Sequence[Token], pages: dict[int, Page], source: str = "ocr") -> list[Provenance]:
    """One entry per page: union of token boxes mapped to ORIGINAL page coordinates."""
    out: list[Provenance] = []
    for pidx in sorted({t.page for t in tokens}):
        box = union([t.bbox for t in tokens if t.page == pidx])
        page = pages.get(pidx)
        orig = _preprocess.map_to_original(page, box) if page is not None else box
        if orig.width <= 0 or orig.height <= 0:
            continue
        out.append(Provenance(page=pidx, bbox=orig.as_list(), source=source))
    return out


def build_field(hit: FieldHit | None, pages: dict[int, Page], params: dict[str, Any],
                source: str = "ocr") -> ReportField:
    if hit is None or (not hit.tokens and not hit.untrusted):
        return ReportField.missing()
    toks = [t for t in hit.tokens if _ALNUM.search(t.text)] or hit.tokens
    value = join_tokens(hit.tokens).strip() or None
    mean_conf = statistics.fmean(t.conf for t in toks) if toks else 0.0
    f = ReportField(value=value, confidence=round(min(1.0, mean_conf * hit.mapping_conf), 4) if value else 0.0,
                    provenance=provenance(hit.tokens, pages, source) if hit.tokens else [])
    if value and mean_conf < params["low_conf_threshold"]:
        f.low_confidence = True
    if hit.untrusted:
        f.untrusted_text = [UntrustedText(text=join_tokens(s), reason=reason,
                                          provenance=provenance(s, pages, source))
                            for s, reason in hit.untrusted]
    return f


@dataclass
class ExtractionResult:
    report: Report
    lines: dict[int, list[Line]]
    meta: dict[str, Any]
    flags: set[str]
    hits: dict[str, FieldHit] = field(default_factory=dict)
    pages: dict[int, Page] = field(default_factory=dict)


def extract_pages(pages: Sequence[Page], words_by_page: dict[int, Sequence[Any]], *,
                  params: dict[str, Any] | None = None, anchors=None,
                  reread: Callable[[Page, BBox], tuple[str, float]] | None = None,
                  tables: dict[int, list[Any]] | None = None) -> ExtractionResult:
    """Run OCR-first mapping over already-OCR'd pages (processed-page word coordinates)."""
    p = {**DEFAULTS, **(params or {})}
    ex = OcrFirstExtractor(p, anchors)
    page_map = {pg.index: pg for pg in pages}
    lines_by_page = {pg.index: page_lines(pg.index, words_by_page.get(pg.index) or []) for pg in pages}
    all_lines = [ln for pg in sorted(page_map) for ln in lines_by_page[pg]]
    hits = ex.map_fields(all_lines, tables)
    fields: dict[str, ReportField] = {}
    for name in ALL_FIELDS:
        hit = hits.get(name)
        if hit is None and name in ("exam_date", "modality"):
            continue
        f = build_field(hit, page_map, p)
        if f.low_confidence and reread is not None and f.provenance:
            f = _reread_field(f, hit, page_map, reread, p)
        fields[name] = f
    report = Report(**fields)
    flags = {"low_confidence"} if any(f.low_confidence for f in fields.values()) else set()
    meta = {"pipeline": "ocr_first", "version": EXTRACT_OCR_VERSION,
            "n_lines": len(all_lines),
            "fields_found": sorted(n for n, f in fields.items() if f.value),
            "fields_missing": sorted(n for n in HEADER_FIELDS + SECTION_FIELDS if not fields[n].value),
            "low_confidence_fields": sorted(n for n, f in fields.items() if f.low_confidence),
            "n_untrusted": sum(len(f.untrusted_text or []) for f in fields.values()),
            "handwriting_reader": "supplied" if reread is not None else "unavailable",
            "n_tables": sum(len(v) for v in (tables or {}).values()),
            "table_sections": {f"{pg}:{ti}": sec for (pg, ti), sec in ex.table_sections.items()}}
    return ExtractionResult(report, lines_by_page, meta, flags, hits, page_map)


def _reread_field(f: ReportField, hit: FieldHit, pages: dict[int, Page],
                  reread: Callable[[Page, BBox], tuple[str, float]], p: dict[str, Any]) -> ReportField:
    """Re-read each page crop of a low-confidence field with a local crop reader."""
    texts, confs = [], []
    for pidx in sorted({t.page for t in hit.tokens}):
        box = union([t.bbox for t in hit.tokens if t.page == pidx])
        txt, c = reread(pages[pidx], box)
        texts.append(txt)
        confs.append(c)
    if not texts or not all(texts):
        return f
    mean = statistics.fmean(confs)
    out = f.model_copy(update={"value": " ".join(texts),
                               "confidence": round(min(1.0, mean * hit.mapping_conf), 4)})
    out.low_confidence = True if mean < p["low_conf_threshold"] else None
    return out


def words_for_pages(ctx: Any, pages: Sequence[Page]) -> dict[int, Sequence[Any]]:
    """OCR words per page (processed coords): from the PHI stage when present, else OCR now."""
    cached = ctx.data.get("ocr_words") or {}
    out: dict[int, Sequence[Any]] = {}
    for pg in pages:
        words = cached.get(pg.index)
        if words is None:
            words = pg.words or None
        if words is None:
            from radintake.tools.pdf_text import page_words

            words = page_words(pg, trace=ctx.trace)
        out[pg.index] = words
    return out


def run(ctx: Any) -> ExtractionResult:
    """OCR-first pipeline over the stage context (used by the extract stage)."""
    t0 = time.perf_counter()
    pages: list[Page] = list(ctx.data.get("pages") or [])
    res = extract_pages(pages, words_for_pages(ctx, pages), params=load_params(ctx.config),
                        tables=ctx.data.get("table_objs"))
    res.meta["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return res
