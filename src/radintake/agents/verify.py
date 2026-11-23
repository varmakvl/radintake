"""Verifier agent: checks an extracted report.v1 and routes doubtful cases to review. It NEVER
changes, replaces or invents a value; it only sets ``grounded`` / ``needs_review`` /
``low_confidence`` on fields and raises case flags.

Checks (PLAN.md 2.5):

* report.v1 schema validation -> ``schema_error``;
* grounding: every non-empty value is fuzzy-matched (rapidfuzz ``partial_ratio``) against the OCR
  text of its cited box plus a margin, then against the whole-document OCR text ->
  ``field.grounded``; an ungrounded value is kept but gets ``needs_review`` +
  ``ungrounded_value``;
* omission: a section heading detected on a page while the field is empty -> ``omission``;
* cross-field sanity: dates parse, IDs match the configured regex, impression is not identical to
  findings (reported as review reasons + ``needs_review`` on the field);
* confidence below ``verify.low_conf_threshold`` -> ``low_confidence`` + ``needs_review``.

The stage is registered as ``verify`` (order 50, after ``extract`` 40). Thresholds live in the
``verify`` section of the config (defaults below). No model or network call is made.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rapidfuzz import fuzz

from radintake.models.report import ALL_FIELDS, SECTION_FIELDS
from radintake.orchestrator.stages import DOCUMENT_TYPES, StageContext, register_stage

VERIFY_VERSION = "verify-1"

DEFAULTS: dict[str, Any] = {
    "ground_min_ratio": 90,       # partial_ratio (0-100) for a value to count as present in the OCR text
    "box_margin_px": 40,          # margin added around a cited box when collecting its OCR words
    "low_conf_threshold": 0.5,    # field confidence below this -> low_confidence + needs_review
    "date_formats": ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d-%b-%Y", "%d %b %Y", "%b %d, %Y",
                     "%B %d, %Y", "%d.%m.%Y", "%Y%m%d", "%m-%d-%Y"],
    "id_patterns": {
        "patient_id": r"^[A-Za-z0-9][A-Za-z0-9._/\-]{1,39}$",
        "accession_number": r"^[A-Za-z0-9][A-Za-z0-9._/\-]{1,39}$",
    },
}
DATE_FIELDS = ("date_of_birth", "exam_date")
ID_FIELDS = ("patient_id", "accession_number")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def load_params(cfg: Any = None) -> dict[str, Any]:
    """Defaults < the ``verify`` section of the run config."""
    raw = getattr(cfg, "raw", None) or {}
    user = raw.get("verify") if isinstance(raw.get("verify"), dict) else {}
    out = {**DEFAULTS, **user}
    out["id_patterns"] = {**DEFAULTS["id_patterns"], **(user.get("id_patterns") or {})}
    return out


def norm(text: str | None) -> str:
    """Lower-case alphanumerics only: robust to OCR spacing / hyphenation / punctuation."""
    return _NON_ALNUM.sub("", (text or "").lower())


@dataclass
class PageOcr:
    """OCR evidence for one page, in ORIGINAL page coordinates."""
    words: list[tuple[list[float], str]] = field(default_factory=list)
    headings: set[str] = field(default_factory=set)  # section names whose heading was detected

    @property
    def text(self) -> str:
        return " ".join(t for _, t in self.words)

    def text_near(self, box: list[float], margin: float) -> str:
        x0, y0, x1, y1 = box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin
        return " ".join(t for b, t in self.words
                        if b[2] >= x0 and b[0] <= x1 and b[3] >= y0 and b[1] <= y1)


@dataclass
class VerifyResult:
    report: dict[str, Any]
    flags: set[str] = field(default_factory=set)
    reasons: list[str] = field(default_factory=list)       # id-free reason codes, e.g. "omission:impression"
    review_fields: list[str] = field(default_factory=list)  # field names that need review

    @property
    def status(self) -> str:
        return "needs_review" if (self.flags or self.reasons) else "ok"


def _parses_as_date(value: str, formats: list[str]) -> bool:
    v = value.strip()
    for fmt in formats:
        try:
            datetime.strptime(v, fmt)
            return True
        except ValueError:
            continue
    return False


def _grounded(value: str, f: dict[str, Any], ocr: dict[int, PageOcr], whole: str, p: dict[str, Any]) -> bool:
    v = norm(value)
    if not v:
        return True  # nothing checkable (punctuation only)
    min_ratio = p["ground_min_ratio"]
    for prov in f.get("provenance") or []:
        po = ocr.get(int(prov["page"]))
        if po is None:
            continue
        near = norm(po.text_near(prov["bbox"], p["box_margin_px"]))
        if near and fuzz.partial_ratio(v, near) >= min_ratio:
            return True
    return bool(whole) and fuzz.partial_ratio(v, whole) >= min_ratio


def verify_report(report: dict[str, Any] | None, ocr: dict[int, PageOcr], *, params: dict[str, Any] | None = None,
                  schema_errors: list[str] | None = None) -> VerifyResult:
    """Verify a report.v1 dict against OCR evidence. Returns a copy with verification marks; values
    are never modified. ``schema_errors`` defaults to validating against ``schemas/report.v1.json``."""
    p = {**DEFAULTS, **(params or {})}
    if report is None:
        return VerifyResult({}, {"schema_error"}, ["schema_error"])
    if schema_errors is None:
        from radintake.agents.extract import report_errors

        schema_errors = report_errors(report)
    if schema_errors:
        return VerifyResult(copy.deepcopy(report), {"schema_error"}, ["schema_error"])

    out = copy.deepcopy(report)
    res = VerifyResult(out)
    whole = norm(" ".join(po.text for _, po in sorted(ocr.items())))
    review: set[str] = set()

    def mark(name: str, flag: str | None, reason: str) -> None:
        out[name]["needs_review"] = True
        review.add(name)
        if flag:
            res.flags.add(flag)
        if reason not in res.reasons:
            res.reasons.append(reason)

    headings = {s for po in ocr.values() for s in po.headings}
    for name in ALL_FIELDS:
        f = out.get(name)
        if f is None:
            continue
        value = f.get("value")
        if value:
            f["grounded"] = _grounded(value, f, ocr, whole, p)
            if not f["grounded"]:
                mark(name, "ungrounded_value", f"ungrounded_value:{name}")
            if f.get("low_confidence") or f.get("confidence", 1.0) < p["low_conf_threshold"]:
                f["low_confidence"] = True
                mark(name, "low_confidence", f"low_confidence:{name}")
            if name in DATE_FIELDS and not _parses_as_date(value, p["date_formats"]):
                mark(name, None, f"invalid_date:{name}")
            if name in ID_FIELDS and not re.match(p["id_patterns"][name], value.strip()):
                mark(name, None, f"invalid_id:{name}")
        elif name in SECTION_FIELDS and name in headings and not f.get("untrusted_text"):
            mark(name, "omission", f"omission:{name}")

    imp, fnd = out.get("impression") or {}, out.get("findings") or {}
    if imp.get("value") and norm(imp["value"]) == norm(fnd.get("value")):
        mark("impression", None, "impression_equals_findings")
    res.review_fields = sorted(review)
    return res


# --------------------------------------------------------------------------- stage wiring
def build_ocr(ctx: StageContext, pages: list[Any]) -> dict[int, PageOcr]:
    """OCR evidence per page mapped to original coordinates, with detected section headings."""
    from radintake.agents import extract_ocr

    words_by_page = extract_ocr.words_for_pages(ctx, pages)
    ex = extract_ocr.OcrFirstExtractor(extract_ocr.load_params(ctx.config))
    out: dict[int, PageOcr] = {}
    for pg in pages:
        words = words_by_page.get(pg.index) or []
        po = PageOcr()
        for w in words:
            txt = (w.text or "").strip()
            if txt:
                po.words.append((pg.to_original(w.bbox).as_list(), txt))
        for ln in extract_ocr.page_lines(pg.index, words):
            h = ex.heading_of(ln)
            if h:
                po.headings.add(h[0])
        out[pg.index] = po
    return out


@register_stage("verify", applies_to=DOCUMENT_TYPES, order=50)
def verify_stage(ctx: StageContext) -> dict[str, Any]:
    report = ctx.data.get("report")
    if report is None:  # extraction failed or broke the contract: already flagged, a human must look
        return {}
    pages = list(ctx.data.get("pages") or [])
    res = verify_report(report, build_ocr(ctx, pages), params=load_params(ctx.config))
    ctx.trace.event("verify", version=VERIFY_VERSION, status=res.status, flags=sorted(res.flags),
                    reasons=res.reasons, fields=res.review_fields)
    out: dict[str, Any] = {"report": res.report, "flags": sorted(res.flags)}
    model = ctx.data.get("report_model")
    if model is not None:
        try:
            from radintake.models.report import Report

            out["report_model"] = Report.from_dict(res.report)
        except Exception:  # noqa: BLE001, S110 - the dict is the contract; the mirror is best effort
            pass
    if res.reasons:
        out["review_reasons"] = [*(ctx.data.get("review_reasons") or []), *res.reasons]
    if res.review_fields:
        out["review_fields"] = sorted({*(ctx.data.get("review_fields") or []), *res.review_fields})
    return out
