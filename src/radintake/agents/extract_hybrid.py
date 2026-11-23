"""Hybrid extraction: OCR-first everywhere, the VLM only where OCR is weak.

Flow (PLAN.md 2.4, hybrid)::

    OCR-first report (words, boxes, section segmentation, header fields - all local)
    (a) section fields that are low_confidence or below extraction.vlm.hybrid_conf_threshold
        -> crop of the PHI-masked page (vlm_client.crop_payload) -> VLM task=crop
           aligned to the OCR words of the crop (fuzz.ratio >= align_min_ratio)
               -> VLM text, OCR boxes, source "hybrid", confidence max(OCR, VLM x ratio)
           not aligned -> VLM text, source "vlm", confidence x unaligned_penalty, low_confidence
    (b) required section fields OCR-first could not find (reading-order / segmentation failure)
        -> VLM task=page on the masked pages, values snapped to OCR tokens (as VLM-first)

When nothing needs the VLM no model is called. When something does but no VLM is usable
(``RADINTAKE_OFFLINE=1`` with no cached local VLM, no API key, calls forbidden, a failed call) the
OCR-first fields are kept and an ``extraction_fallback`` trace event records why.
"""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from radintake.agents import extract_ocr, extract_vlm
from radintake.agents.extract_ocr import ExtractionResult, OcrFirstExtractor, join_tokens, union
from radintake.models.report import SECTION_FIELDS, Report, ReportField
from radintake.tools.vlm_client import VlmClient, crop_payload, select_vlm

EXTRACT_HYBRID_VERSION = "extract-hybrid-1"


def _weak_fields(report: Report, threshold: float) -> list[str]:
    out = []
    for n in SECTION_FIELDS:
        f: ReportField = getattr(report, n)
        if f.value and (f.low_confidence or f.confidence < threshold):
            out.append(n)
    return out


def reread_crop(client: VlmClient, ctx: Any, res: ExtractionResult, name: str,
                payloads: dict[int, Any], p: dict[str, Any]) -> ReportField:
    """Re-read one weak field crop by crop; keep OCR boxes, mark the source."""
    hit = res.hits[name]
    old: ReportField = getattr(res.report, name)
    prompt = extract_vlm.render_prompt("crop")
    texts, confs, ratios = [], [], []
    for pidx in sorted({t.page for t in hit.tokens}):
        toks = [t for t in hit.tokens if t.page == pidx]
        crop = crop_payload(payloads[pidx], union([t.bbox for t in toks]))
        resp = client.call(crop, prompt_id=extract_vlm.PROMPT_ID, prompt=prompt, trace=ctx.trace, task="crop")
        data = resp.data if isinstance(resp.data, dict) else {}
        text = str(data.get("text") or "").strip()
        if not text:
            return old
        texts.append(text)
        confs.append(float(data.get("confidence", p["vlm_confidence"]) or 0.0))
        ratios.append(fuzz.ratio(extract_vlm._norm(text), extract_vlm._norm(join_tokens(toks))))
    value = " ".join(texts)
    vault = ctx.data.get("phi_vault")
    if vault is not None:
        value = vault.restore(value)
    vconf = min(1.0, max(0.0, sum(confs) / len(confs)))
    ratio = min(ratios)
    if ratio >= p["align_min_ratio"]:
        prov = [pv.model_copy(update={"source": "hybrid"}) for pv in old.provenance]
        return old.model_copy(update={"value": value, "provenance": prov, "low_confidence": None,
                                      "confidence": round(max(old.confidence, vconf * ratio / 100.0), 4)})
    prov = [pv.model_copy(update={"source": "vlm"}) for pv in old.provenance]
    return old.model_copy(update={"value": value, "provenance": prov, "low_confidence": True,
                                  "confidence": round(vconf * p["unaligned_penalty"], 4)})


def run(ctx: Any, client: VlmClient | None = None, page_client: VlmClient | None = None) -> ExtractionResult:
    """Hybrid pipeline over the stage context (used by the extract stage)."""
    res = extract_ocr.run(ctx)
    res.meta.update(pipeline="hybrid", version=EXTRACT_HYBRID_VERSION, vlm_calls=0)
    p = extract_vlm.params(ctx.config)
    weak = _weak_fields(res.report, p["hybrid_conf_threshold"])
    missing = [n for n in SECTION_FIELDS if not getattr(res.report, n).value]
    res.meta.update(vlm_crop_fields=weak, vlm_repair_fields=missing)
    if not weak and not missing:
        return res  # OCR is confident everywhere: no model call needed
    payload_list = list(ctx.data.get("phi_masked_pages") or [])
    if not payload_list:
        return extract_vlm.fallback(ctx, res, "hybrid", "no_phi_masked_payload")
    payloads = {int(pl.data.get("page") or i + 1): pl for i, pl in enumerate(payload_list)}

    fields = {n: getattr(res.report, n) for n in Report.model_fields if getattr(res.report, n) is not None}
    reasons: list[str] = []
    calls = 0
    if weak:
        c, why = (client, "injected") if client is not None else select_vlm(ctx.config, ctx.trace, task="crop")
        if c is None:
            reasons.append(why)
        else:
            for name in weak:
                try:
                    fields[name] = reread_crop(c, ctx, res, name, payloads, p)
                    calls += 1
                except Exception as exc:  # noqa: BLE001 - keep the OCR field
                    reasons.append(f"vlm_error:{type(exc).__name__}")
    if missing:
        c, why = ((page_client or client, "injected") if (page_client or client) is not None
                  else select_vlm(ctx.config, ctx.trace, task="page"))
        if c is None:
            reasons.append(why)
        else:
            try:
                got = extract_vlm.page_fields(c, payload_list, ctx)
                calls += len(payload_list)
                extractor = OcrFirstExtractor(extract_ocr.load_params(ctx.config))
                tokens = extract_vlm.all_tokens(res)
                for name in missing:
                    items = got.get(name) or []
                    if items:
                        fields[name] = extract_vlm.vlm_text_field(
                            " ".join(v for _, v, _ in items), [(pg, b) for pg, _, b in items], tokens,
                            res.pages, extractor, p, ctx.data.get("phi_vault"))
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"vlm_error:{type(exc).__name__}")
    for why in dict.fromkeys(reasons):
        extract_vlm.fallback(ctx, res, "hybrid", why)
    report = Report(**fields)
    flags = {"low_confidence"} if any(getattr(report, n) and getattr(report, n).low_confidence
                                      for n in Report.model_fields) else set()
    res.meta.update(vlm_calls=calls)
    return ExtractionResult(report, res.lines, res.meta, flags, res.hits, res.pages)
