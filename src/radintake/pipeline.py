"""Per-file pipeline behind :class:`radintake.sdk.Pipeline` (internal: not covered by the
API stability policy in ``docs/api-stability.md``; use ``radintake.process`` / ``Pipeline``).

``process_file(path, out_dir)`` routes the file (fixed step), runs the stage plan for its
input type under the planner budgets, writes ``<out_dir>/<input-name>.json`` validating
against ``schemas/cli_output.v1.json`` and returns that document. It never raises for a bad
input: failures become ``status: "error"`` documents.

Document inputs get their ``report`` from the ``extract`` stage
(:mod:`radintake.agents.extract`); a document with no report is ``needs_review``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from radintake.config import Config, load_config, schema_path
from radintake.logging import get_logger
from radintake.orchestrator import stages as stage_registry
from radintake.orchestrator.planner import Planner
from radintake.orchestrator.router import ROUTER_VERSION, route
from radintake.orchestrator.stages import DOCUMENT_TYPES, StageContext
from radintake.tools import ocr as ocr_tool
from radintake.tracing.writer import CaseTrace, TraceWriter

SCHEMA_VERSION = "cli_output.v1"
ALLOWED_FLAGS = frozenset({"low_confidence", "ungrounded_value", "omission", "schema_error",
                           "prompt_injection_suspected", "phi_uncertain", "preprocess_failed", "timeout"})

log = get_logger("radintake.pipeline")
_validator = None


def _schema_validator():
    global _validator
    if _validator is None:
        from jsonschema import Draft202012Validator

        p = schema_path("cli_output.v1.json")
        _validator = Draft202012Validator(json.loads(p.read_text(encoding="utf-8"))) if p.is_file() else False
    return _validator or None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def output_path(out_dir: Path, input_name: str) -> Path:
    return Path(out_dir) / f"{input_name}.json"


def _base_doc(input_name: str, sha: str, input_type: str, trace_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "input": input_name,
        "input_sha256": sha,
        "input_type": input_type,
        "status": "error",
        "trace_id": trace_id,
        "error": None,
        "n_pages": 0,
        "report": None,
        "phi": None,
        "dicom": None,
        "flags": [],
    }


def write_doc(doc: dict[str, Any], out_dir: Path) -> Path:
    p = output_path(out_dir, doc["input"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return p


def _finalise(doc: dict[str, Any], trace: CaseTrace | None) -> dict[str, Any]:
    """Keep only schema flags, validate; a schema failure degrades to an error document."""
    doc["flags"] = sorted(f for f in set(doc.get("flags") or []) if f in ALLOWED_FLAGS)
    v = _schema_validator()
    if v is not None:
        errors = list(v.iter_errors(doc))
        if errors:
            fields = sorted({".".join(str(x) for x in e.absolute_path) or "<root>" for e in errors})
            safe = _base_doc(doc["input"], doc["input_sha256"], doc["input_type"], doc["trace_id"])
            safe.update(status="error", n_pages=int(doc.get("n_pages") or 0), flags=["schema_error"],
                        error=f"output failed schema validation at: {', '.join(fields)[:300]}")
            if trace is not None:
                trace.event("schema_error", fields=fields)
            doc = safe
    return doc


def process_file(path: str | Path, out_dir: str | Path, *, config: Config | None = None,
                 trace_writer: TraceWriter | None = None, input_name: str | None = None,
                 write: bool = True) -> dict[str, Any]:
    """Process one input file and (by default) write its JSON. Never raises for bad inputs."""
    path, out_dir = Path(path), Path(out_dir)
    cfg = config or load_config()
    ocr_tool.set_engine(cfg.ocr_engine)
    name = input_name or path.name
    if trace_writer is None:
        trace_writer = TraceWriter(cfg.trace_dir or (out_dir / "_traces"), model_ids=cfg.model_ids,
                                   offline=cfg.offline)
    stage_cfg = cfg.raw.get("stages") if isinstance(cfg.raw.get("stages"), dict) else {}
    stage_registry.discover(stage_cfg.get("plugins") or [])
    t_start = time.monotonic()
    try:
        sha = sha256_file(path)
    except OSError as exc:
        sha = hashlib.sha256(b"").hexdigest()
        trace = trace_writer.start_case(name, sha)
        doc = _base_doc(name, sha, "unknown", trace.trace_id)
        doc["error"] = f"cannot read input: {type(exc).__name__}"
        return _emit(doc, trace, out_dir, write)

    trace = trace_writer.start_case(name, sha)
    doc = _base_doc(name, sha, "unknown", trace.trace_id)
    try:
        t0 = time.perf_counter()
        r = route(path)
        trace.tool_call("router", {}, {"input_type": r.input_type, "n_pages": r.n_pages,
                                       "signals": r.signals, "error": r.error},
                        round((time.perf_counter() - t0) * 1000.0, 3), tool_version=ROUTER_VERSION)
        doc["input_type"] = r.input_type
        if not r.ok:
            doc["error"] = r.error or "unsupported input"
            return _emit(doc, trace, out_dir, write)

        ctx = StageContext(path=path, input_name=name, input_type=r.input_type, input_sha256=sha,
                           out_dir=out_dir, config=cfg, trace=trace,
                           deadline=t_start + cfg.budgets.case_deadline_s,
                           data={"n_pages": r.n_pages or 0, "route": r})
        plan = stage_registry.plan_for(r.input_type)
        outcome = Planner(cfg).run(ctx, plan)

        doc["status"] = outcome.status
        doc["error"] = outcome.error
        doc["n_pages"] = int(ctx.data.get("n_pages") or 0)
        doc["flags"] = sorted(ctx.flags)
        for key in ("report", "phi", "dicom"):
            doc[key] = ctx.data.get(key)
        if ctx.data.get("tables"):
            doc["tables"] = ctx.data["tables"]
        if doc["status"] == "ok":
            missing = doc["report"] is None if r.input_type in DOCUMENT_TYPES else doc["dicom"] is None
            if missing:  # nothing produced the main result yet -> a human must look
                doc["status"] = "needs_review"
        if doc["status"] == "needs_review":
            _enqueue_review(doc, ctx, cfg, out_dir, write)
    except Exception as exc:  # noqa: BLE001 - last line of defence: never crash the batch
        doc["status"] = "error"
        doc["error"] = f"pipeline failure: {type(exc).__name__}"
        trace.event("error", error_type=type(exc).__name__)
    return _emit(doc, trace, out_dir, write)


def _enqueue_review(doc: dict[str, Any], ctx: StageContext, cfg: Config, out_dir: Path, write: bool) -> None:
    """Persist a needs_review case: ids, flags, reason codes and field NAMES only (no PHI/values)."""
    try:
        from radintake.store import review_queue

        reasons = [str(r).split(":", 1)[0].strip() for r in ctx.data.get("review_reasons") or []]  # codes only
        item = review_queue.add_item(doc["input"], flags=list(ctx.flags), reasons=reasons,
                                     fields=list(ctx.data.get("review_fields") or []),
                                     trace_id=doc["trace_id"], data_dir=cfg.data_dir)
        if write:
            review_queue.append_jsonl(out_dir, item)
    except Exception as exc:  # noqa: BLE001 - the queue must never break a case
        log.warning("review_queue_failed", error_type=type(exc).__name__)


def _emit(doc: dict[str, Any], trace: CaseTrace, out_dir: Path, write: bool) -> dict[str, Any]:
    doc = _finalise(doc, trace)
    if write:
        write_doc(doc, out_dir)
    trace.close(doc["status"], input_type=doc["input_type"], flags=doc["flags"], n_pages=doc["n_pages"])
    log.info("case_done", trace_id=trace.trace_id, input_sha256=doc["input_sha256"][:12],
             input_type=doc["input_type"], status=doc["status"], flags=doc["flags"])
    return doc
