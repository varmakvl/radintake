"""Extraction stage: runs the configured extraction pipeline and emits a report.v1 block.

Registered as stage ``extract`` (order 40: after ``preprocess`` 20 and ``phi`` 30) for every
document input. The pipeline is chosen by config ``extraction.pipeline``; each pipeline is a
callable ``run(ctx) -> ExtractionResult`` registered in :data:`PIPELINES`.

The stage returns ``report`` (the schema-shaped dict written to the CLI JSON, validated against
``schemas/report.v1.json``), the in-process ``report_model`` (:class:`~radintake.models.report.Report`)
and ``extraction`` metadata (counts / field names only) for the verifier. The trace gets one
``extraction`` event with per-field value HASHES, confidences and provenance counts - never text.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from radintake.config import REPO_ROOT
from radintake.orchestrator.stages import DOCUMENT_TYPES, StageContext, register_stage

DEFAULT_PIPELINE = "ocr_first"


def _ocr_first(ctx: StageContext):
    from radintake.agents import extract_ocr

    return extract_ocr.run(ctx)


PIPELINES: dict[str, Callable[[StageContext], Any]] = {"ocr_first": _ocr_first}


def register_pipeline(name: str, fn: Callable[[StageContext], Any]) -> None:
    PIPELINES[name] = fn


def pipeline_name(cfg: Any) -> str:
    raw = getattr(cfg, "raw", None) or {}
    ext = raw.get("extraction") if isinstance(raw.get("extraction"), dict) else {}
    return str(ext.get("pipeline") or DEFAULT_PIPELINE)


@lru_cache(maxsize=1)
def report_validator():
    from jsonschema import Draft202012Validator

    schema = json.loads((REPO_ROOT / "schemas" / "report.v1.json").read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def report_errors(report: dict[str, Any]) -> list[str]:
    return sorted(".".join(str(x) for x in e.absolute_path) or "<root>"
                  for e in report_validator().iter_errors(report))


@register_stage("extract", applies_to=DOCUMENT_TYPES, order=40)
def extract_stage(ctx: StageContext) -> dict[str, Any]:
    name = pipeline_name(ctx.config)
    fn = PIPELINES.get(name)
    if fn is None:
        ctx.trace.event("extraction_pipeline_unknown", mode=name, decision="ocr_first")
        name, fn = DEFAULT_PIPELINE, PIPELINES[DEFAULT_PIPELINE]
    res = fn(ctx)
    report = res.report.to_dict()
    flags = set(res.flags)
    errors = report_errors(report)
    if errors:  # never emit a report that breaks the contract
        flags.add("schema_error")
        ctx.trace.event("extraction_schema_error", fields=errors)
    ctx.trace.event(
        "extraction",
        mode=str(res.meta.get("pipeline", name)),
        version=str(res.meta.get("version", "")),
        fields={k: {"value": v["value"], "confidence": v["confidence"], "n_provenance": len(v["provenance"]),
                    "pages": sorted({p["page"] for p in v["provenance"]}),
                    "n_untrusted": len(v.get("untrusted_text") or []),
                    "low_confidence": bool(v.get("low_confidence"))}
                for k, v in report.items()},
        meta={k: v for k, v in res.meta.items() if k not in ("pipeline", "version")},
        flags=sorted(flags),
    )
    out: dict[str, Any] = {"report": None if errors else report, "report_model": res.report,
                           "extraction": res.meta, "flags": sorted(flags)}
    if not any(v["value"] for v in report.values()):  # nothing readable: a human must look
        out["review_reasons"] = [*(ctx.data.get("review_reasons") or []), "extract: no fields found"]
    return out
