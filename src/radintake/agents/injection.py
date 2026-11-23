"""Prompt-injection detector. Report text is DATA: instruction-like text found in extracted
free-text fields or in the page OCR text raises ``prompt_injection_suspected`` (case ->
needs_review) and marks the affected fields ``needs_review``. It never removes, rewrites or obeys
the text - the value stays exactly as extracted. Local patterns + a small word-count heuristic;
no model or network call.

Registered as stage ``injection`` (order 55, after ``verify`` 50).
"""

from __future__ import annotations

import re
from typing import Any

from radintake.agents.extract_ocr import DEFAULT_UNTRUSTED_PATTERNS
from radintake.models.report import SECTION_FIELDS
from radintake.orchestrator.stages import DOCUMENT_TYPES, StageContext, register_stage

INJECTION_VERSION = "injection-1"

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


def compile_patterns(extra: list[str] | None = None) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in [*_ALL, *(extra or [])]]


def looks_injected(text: str | None, patterns: list[re.Pattern[str]]) -> bool:
    if not text:
        return False
    flat = re.sub(r"\s+", " ", text)
    return any(rx.search(text) or rx.search(flat) for rx in patterns)


def detect_in_report(report: dict[str, Any], patterns: list[re.Pattern[str]]) -> list[str]:
    """Names of fields whose value or untrusted_text holds instruction-like text."""
    hit: list[str] = []
    for name, f in report.items():
        if f.get("untrusted_text") or looks_injected(f.get("value"), patterns):
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
    fields = detect_in_report(report, patterns) if report else []
    page_hit = any(looks_injected(t, patterns) for t in _page_texts(ctx))
    if not fields and not page_hit:
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
    ctx.trace.event("injection", version=INJECTION_VERSION, fields=fields, page_text=page_hit)
    return out
