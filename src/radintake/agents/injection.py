"""Prompt-injection detector. Report text is DATA: instruction-like text found in extracted
free-text fields or in the page OCR text raises ``prompt_injection_suspected`` (case ->
needs_review) and marks the affected fields ``needs_review``. It never removes, rewrites or obeys
the text - the value stays exactly as extracted. Local patterns, an OCR-tolerant fuzzy phrase matcher and the PDF hidden-text signal;
no model or network call.

Registered as stage ``injection`` (order 55, after ``verify`` 50).
"""

from __future__ import annotations

import re
from typing import Any

from radintake.agents.extract_ocr import DEFAULT_UNTRUSTED_PATTERNS
from radintake.models.report import SECTION_FIELDS
from radintake.orchestrator.stages import DOCUMENT_TYPES, StageContext, register_stage

INJECTION_VERSION = "injection-2"

EXTRA_PATTERNS: list[str] = [
    r"\b(?:ignore|disregard|forget)\b.{0,30}\b(?:instructions?|prompts?|rules?)\b",
    r"\bnew (?:instructions?|task|rules?)\s*:",
    r"\b(?:system|assistant|developer)\s*(?:prompt|message)\b",
    r"\b(?:do not|don't)\s+(?:flag|mention|report|review)\b",
    r"\b(?:report|state|write|say|answer)\b.{0,25}\b(?:normal|no abnormalit(?:y|ies))\b.{0,20}\b(?:regardless|always|only)\b",
    r"\bas an? (?:ai|language model|assistant)\b",
    r"<\s*/?\s*(?:system|instructions?)\s*>",
]
_ALL = [*DEFAULT_UNTRUSTED_PATTERNS, *EXTRA_PATTERNS]

# --- OCR-robust second pass: normalise OCR confusions, then fuzzy-match canonical phrases -------------
CANONICAL_PHRASES: tuple[str, ...] = (
    "ignore previous instructions", "ignore all previous instructions", "disregard prior instructions",
    "forget your instructions", "new instructions", "system prompt", "do not flag this report",
    "report as normal regardless", "as an ai language model",
)
DEFAULT_FUZZY_THRESHOLD = 86.0  # rapidfuzz partial_ratio (0-100) of the squashed text vs a squashed phrase
_CONFUSABLE = str.maketrans({"1": "l", "|": "l", "!": "l", "i": "l", "0": "o", "5": "s", "$": "s", "8": "b"})


def normalise_ocr(text: str) -> str:
    """Lowercase, fold common OCR confusions (l/1/i/|, 0/o, rn/m, vv/w) and drop spacing/punctuation, so
    split words ('ign ore'), merged words and character swaps compare on equal footing."""
    t = re.sub(r"[^a-z0-9|!$]", "", text.lower())
    return t.translate(_CONFUSABLE).replace("rn", "m").replace("vv", "w")


_PHRASES_N: tuple[str, ...] = tuple(normalise_ocr(p) for p in CANONICAL_PHRASES)


def fuzzy_injected(text: str | None, threshold: float = DEFAULT_FUZZY_THRESHOLD) -> bool:
    """True when a canonical instruction phrase appears in ``text`` up to OCR damage."""
    if not text:
        return False
    from rapidfuzz import fuzz

    flat = normalise_ocr(text)
    for ph in _PHRASES_N:
        if len(flat) < len(ph) * 0.7:
            continue
        al = fuzz.partial_ratio_alignment(ph, flat)
        # the WHOLE phrase must align (a partial overlap at the edge, e.g. "previous instructions" for
        # "ignore previous instructions", is ordinary clinical wording, not an instruction)
        if al is not None and al.score >= threshold \
                and al.dest_end - al.dest_start >= 0.85 * len(ph):
            return True
    return False


def compile_patterns(extra: list[str] | None = None) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in [*_ALL, *(extra or [])]]


def looks_injected(text: str | None, patterns: list[re.Pattern[str]],
                   fuzzy_threshold: float | None = DEFAULT_FUZZY_THRESHOLD) -> bool:
    """Regex patterns first; then the OCR-tolerant phrase matcher (disabled when threshold is None)."""
    if not text:
        return False
    flat = re.sub(r"\s+", " ", text)
    if any(rx.search(text) or rx.search(flat) for rx in patterns):
        return True
    return fuzzy_threshold is not None and fuzzy_injected(text, fuzzy_threshold)


def detect_in_report(report: dict[str, Any], patterns: list[re.Pattern[str]],
                     fuzzy_threshold: float | None = DEFAULT_FUZZY_THRESHOLD) -> list[str]:
    """Names of fields whose value or untrusted_text holds instruction-like text (values never edited)."""
    hit: list[str] = []
    for name, f in report.items():
        if f.get("untrusted_text") or looks_injected(f.get("value"), patterns, fuzzy_threshold):
            hit.append(name)
    return sorted(hit)


def _page_texts(ctx: StageContext) -> list[str]:
    from radintake.agents import extract_ocr

    pages = list(ctx.data.get("pages") or [])
    words = extract_ocr.words_for_pages(ctx, pages)
    return ["\n".join(ln.text for ln in extract_ocr.page_lines(pg.index, words.get(pg.index) or []))
            for pg in pages]


@register_stage("injection", applies_to=DOCUMENT_TYPES, order=55)
def injection_stage(ctx: StageContext) -> dict[str, Any]:
    raw = getattr(ctx.config, "raw", None) or {}
    cfg = raw.get("injection") if isinstance(raw.get("injection"), dict) else {}
    patterns = compile_patterns(cfg.get("extra_patterns"))
    report = ctx.data.get("report")
    thr = cfg.get("fuzzy_threshold", DEFAULT_FUZZY_THRESHOLD)
    thr = None if thr is None else float(thr)
    fields = detect_in_report(report, patterns, thr) if report else []
    page_hit = any(looks_injected(t, patterns, thr) for t in _page_texts(ctx))
    # Decision: ANY text the loader dropped as invisible (white / tiny / off-page, PDF text layer) is a
    # signal - legitimate digital reports do not carry invisible text. Counts only, never the text.
    hidden = sum((pg.meta.get("text_layer") or {}).get("hidden_words", 0)
                 for pg in (ctx.data.get("pages") or []) if getattr(pg, "meta", None))
    if not fields and not page_hit and not hidden:
        return {}
    out: dict[str, Any] = {"flags": ["prompt_injection_suspected"],
                           "review_reasons": [*(ctx.data.get("review_reasons") or []),
                                              "prompt_injection_suspected"]}
    if report:
        marked = {n: dict(v) for n, v in report.items()}
        names = fields or [n for n in SECTION_FIELDS if marked.get(n, {}).get("value")]
        for n in names:
            marked[n]["needs_review"] = True  # values untouched: the text stays data
        out["report"] = marked
        out["review_fields"] = sorted({*(ctx.data.get("review_fields") or []), *names})
    ctx.trace.event("injection", version=INJECTION_VERSION, fields=fields, page_text=page_hit, hidden_words=hidden)
    return out
