"""VLM-first extraction: PHI-masked page image -> VLM JSON -> values snapped to OCR word boxes.

Flow (PLAN.md 2.4, VLM-first)::

    OCR-first run on the unmasked page (local)  -> header + exam fields, OCR tokens for snapping
    for each page: the PHI agent's MaskedPayload (phi_masked=True) -> VLM (prompts/extraction_v1.md,
                   task=page, JSON-constrained) -> section values + approximate boxes
    per section: placeholders restored locally from the PHI vault; instruction-like sentences moved
                 to untrusted_text; value fuzzy-aligned to the OCR token stream (rapidfuzz):
                 aligned   -> box = union of the matched OCR word boxes, source "hybrid"
                 unaligned -> the VLM's approximate box, source "vlm", confidence x penalty,
                              low_confidence

Header fields (name, IDs, dates) ALWAYS come from local OCR - never from the API, which only ever
sees the masked page. With no usable VLM (``RADINTAKE_OFFLINE=1`` and no local page-level VLM, no
API key, model calls forbidden, or a failed call) the pipeline falls back to the OCR-first report
and records an ``extraction_fallback`` trace event with the reason.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from rapidfuzz import fuzz

from radintake.agents import extract_ocr
from radintake.agents import preprocess as _preprocess
from radintake.agents.extract_ocr import ExtractionResult, OcrFirstExtractor, Token
from radintake.core.types import BBox, Page
from radintake.models.report import SECTION_FIELDS, Provenance, Report, ReportField, UntrustedText
from radintake.tools.vlm_client import VlmClient, load_prompt, select_vlm

EXTRACT_VLM_VERSION = "extract-vlm-1"
PROMPT_ID = "extraction_v1"

DEFAULTS: dict[str, Any] = {
    "align_min_ratio": 80,      # fuzz.ratio (0-100) for a VLM value to count as aligned to OCR tokens
    "vlm_confidence": 0.9,      # prior confidence of a VLM transcription
    "unaligned_penalty": 0.5,   # confidence multiplier for a VLM value with no OCR support
    "hybrid_conf_threshold": 0.6,  # hybrid: OCR field confidence below this -> VLM crop re-read
}


def params(cfg: Any) -> dict[str, Any]:
    raw = getattr(cfg, "raw", None) or {}
    ext = raw.get("extraction") if isinstance(raw.get("extraction"), dict) else {}
    v = ext.get("vlm") if isinstance(ext.get("vlm"), dict) else {}
    return {**DEFAULTS, **{k: v[k] for k in DEFAULTS if k in v}}


def render_prompt(task: str) -> str:
    return load_prompt(PROMPT_ID).replace("{task}", task)


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


def snap(value: str, tokens: Sequence[Token], min_ratio: float) -> tuple[list[Token], float]:
    """Contiguous OCR token window whose text best matches ``value`` (fuzz.ratio, normalised)."""
    target = _norm(value)
    if not target or not tokens:
        return [], 0.0
    norms = [_norm(t.text) for t in tokens]
    best: tuple[int, int, float] = (0, 0, 0.0)
    for i in range(len(tokens)):
        if not norms[i]:
            continue
        text = ""
        for j in range(i, len(tokens)):
            if norms[j]:
                text = f"{text} {norms[j]}" if text else norms[j]
            if len(text) > 1.3 * len(target) + 5:
                break
            r = fuzz.ratio(text, target)
            if r > best[2]:
                best = (i, j + 1, r)
    i, j, r = best
    return (list(tokens[i:j]), r) if r >= min_ratio else ([], r)


def split_sentences(text: str, extractor: OcrFirstExtractor) -> tuple[str, list[str]]:
    """(clinical text, instruction-like sentences kept verbatim)."""
    parts = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    kept = [s for s in parts if not any(rx.search(s) for rx in extractor.untrusted)]
    bad = [s for s in parts if s not in kept]
    return " ".join(kept), bad


def _prov(tokens: Sequence[Token], pages: dict[int, Page], source: str) -> list[Provenance]:
    return extract_ocr.provenance(tokens, pages, source)


def _approx_prov(boxes: Sequence[tuple[int, Any]], pages: dict[int, Page]) -> list[Provenance]:
    """The VLM's own approximate boxes (processed-page coords) mapped to original coords."""
    out: list[Provenance] = []
    for pidx, raw in boxes:
        try:
            b = BBox.from_seq(raw)
        except (TypeError, ValueError):
            continue
        page = pages.get(pidx)
        if page is None:
            continue
        w, h = page.size
        b = b.clip(w, h)
        orig = _preprocess.map_to_original(page, b)
        if orig.width > 0 and orig.height > 0:
            out.append(Provenance(page=pidx, bbox=orig.as_list(), source="vlm"))
    return out


def vlm_text_field(value: str, boxes: Sequence[tuple[int, Any]], tokens: Sequence[Token],
                   pages: dict[int, Page], extractor: OcrFirstExtractor, p: dict[str, Any],
                   vault: Any = None) -> ReportField:
    """A section field from VLM text: placeholders restored, untrusted sentences split, snapped."""
    if vault is not None:
        value = vault.restore(value)
    clinical, bad = split_sentences(value, extractor)
    if not clinical and not bad:
        return ReportField.missing()
    f = ReportField.missing()
    if clinical:
        toks, r = snap(clinical, tokens, p["align_min_ratio"])
        if toks:
            f = ReportField(value=clinical, confidence=round(p["vlm_confidence"] * r / 100.0, 4),
                            provenance=_prov(toks, pages, "hybrid"))
        else:
            f = ReportField(value=clinical,
                            confidence=round(p["vlm_confidence"] * p["unaligned_penalty"], 4),
                            provenance=_approx_prov(boxes, pages), low_confidence=True)
    if bad:
        uts = []
        for s in bad:
            toks, _ = snap(s, tokens, p["align_min_ratio"])
            uts.append(UntrustedText(text=s, reason="instruction_like_text",
                                     provenance=_prov(toks, pages, "hybrid") if toks else _approx_prov(boxes, pages)))
        f.untrusted_text = uts
    return f


def all_tokens(res: ExtractionResult) -> list[Token]:
    return [t for pidx in sorted(res.lines) for ln in res.lines[pidx] for t in ln.tokens]


def fallback(ctx: Any, res: ExtractionResult, mode: str, reason: str) -> ExtractionResult:
    """OCR-first result standing in for ``mode``; the decision is recorded in the trace."""
    ctx.trace.event("extraction_fallback", mode=mode, decision="ocr_first", reason=reason)
    res.meta.update(pipeline=mode, fallback="ocr_first", fallback_reason=reason)
    return res


def page_fields(client: VlmClient, payloads: Sequence[Any], ctx: Any) -> dict[str, list[tuple[int, str, Any]]]:
    """Call the VLM once per masked page: section -> [(page, value, approx bbox)]."""
    prompt = render_prompt("page")
    out: dict[str, list[tuple[int, str, Any]]] = {}
    for pl in payloads:
        page_idx = int((getattr(pl, "data", None) or {}).get("page") or pl.meta.get("page") or 1)
        resp = client.call(pl, prompt_id=PROMPT_ID, prompt=prompt, trace=ctx.trace, task="page")
        fields = (resp.data or {}).get("fields") if isinstance(resp.data, dict) else None
        for name in SECTION_FIELDS:
            item = (fields or {}).get(name)
            if isinstance(item, dict) and isinstance(item.get("value"), str) and item["value"].strip():
                out.setdefault(name, []).append((page_idx, item["value"].strip(), item.get("bbox")))
    return out


def run(ctx: Any, client: VlmClient | None = None) -> ExtractionResult:
    """VLM-first pipeline over the stage context (used by the extract stage)."""
    res = extract_ocr.run(ctx)  # local: header fields + OCR tokens for snapping
    payloads = list(ctx.data.get("phi_masked_pages") or [])
    if not payloads:
        return fallback(ctx, res, "vlm_first", "no_phi_masked_payload")
    reason = "injected"
    if client is None:
        client, reason = select_vlm(ctx.config, ctx.trace, task="page")
    if client is None:
        return fallback(ctx, res, "vlm_first", reason)
    p = params(ctx.config)
    try:
        got = page_fields(client, payloads, ctx)
    except Exception as exc:  # noqa: BLE001 - a failed/refused call never loses the OCR result
        return fallback(ctx, res, "vlm_first", f"vlm_error:{type(exc).__name__}")
    extractor = OcrFirstExtractor(extract_ocr.load_params(ctx.config))
    tokens = all_tokens(res)
    vault = ctx.data.get("phi_vault")
    fields = {n: getattr(res.report, n) for n in Report.model_fields if getattr(res.report, n) is not None}
    for name in SECTION_FIELDS:
        items = got.get(name) or []
        if not items:
            fields[name] = ReportField.missing()
            continue
        value = " ".join(v for _, v, _ in items)
        fields[name] = vlm_text_field(value, [(pg, b) for pg, _, b in items], tokens, res.pages,
                                      extractor, p, vault)
    report = Report(**fields)
    flags = {"low_confidence"} if any(getattr(report, n) and getattr(report, n).low_confidence
                                      for n in Report.model_fields) else set()
    meta = {**res.meta, "pipeline": "vlm_first", "version": EXTRACT_VLM_VERSION, "vlm_backend": reason,
            "vlm_model": client.model_id, "vlm_calls": len(payloads),
            "unaligned_fields": sorted(n for n in SECTION_FIELDS
                                       if any(pv.source == "vlm" for pv in fields[n].provenance))}
    return ExtractionResult(report, res.lines, meta, flags, res.hits, res.pages)


__all__ = [
    "PROMPT_ID",
    "all_tokens",
    "fallback",
    "page_fields",
    "params",
    "render_prompt",
    "run",
    "snap",
    "vlm_text_field",
]
