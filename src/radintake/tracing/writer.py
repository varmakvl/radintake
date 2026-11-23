"""Replayable, PHI-free run traces.

Layout::

    <trace_dir>/<run_id>/manifest.json      run manifest (schemas/trace.v1.json#/$defs/manifest)
    <trace_dir>/<run_id>/<case_id>.jsonl    one event per line (schemas/trace.v1.json)

PHI policy (requirements.md constraint 3 - prefer hashes): every string value in an event
payload is replaced by ``"sha256:<hex>"`` unless its key is in :data:`SAFE_KEYS` (ids, stage
and tool names, statuses, model ids, prompt hashes...). Even safe values are passed through
the PHI scrubber of :mod:`radintake.logging`. So report text and field values never reach a
trace; re-scoring compares stored per-field value hashes and scores instead.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import threading
import time
import uuid
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from radintake import __version__
from radintake.logging import scrub

TRACE_SCHEMA_VERSION = "trace.v1"
MANIFEST_SCHEMA_VERSION = "trace_manifest.v1"

# Keys whose string values are identifiers / enums, safe to keep verbatim (still scrubbed).
SAFE_KEYS = frozenset({
    "kind", "stage", "stages", "plan", "agent", "tool", "tool_version", "model_id", "model_kind",
    "prompt_id", "prompt_hash", "payload_hash", "output_hash", "status", "input_type", "flag",
    "flags", "field", "fields", "error_type", "error_code", "case_id", "run_id", "trace_id",
    "input_sha256", "sha256", "source", "version", "schema_version", "decision", "action", "op",
    "ops", "reason", "outcome", "error", "method", "backend", "device", "dtype", "mode",
})
_MAX_SAFE_LEN = 300

_TOOL_DISTS = ("radintake", "numpy", "opencv-python-headless", "Pillow", "pydicom", "pypdfium2",
               "rapidfuzz", "pytesseract", "jsonschema", "structlog", "pydantic")


def hash_text(text: str) -> str:
    """Stable value hash used in traces: ``sha256:<64 hex>`` of the UTF-8 text."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sanitize(obj: Any, key: str | None = None, *, _safe: bool = False) -> Any:
    """Make a payload trace-safe: hash every string not under a SAFE_KEYS key."""
    safe = _safe or (key in SAFE_KEYS)
    if isinstance(obj, str):
        if safe:
            return scrub(obj)[:_MAX_SAFE_LEN]
        return hash_text(obj)
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, dict):
        # safety never propagates into nested dicts: each value is judged by its own key
        return {_dict_key(k): sanitize(v, str(k)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v, key, _safe=safe) for v in obj]
    if isinstance(obj, np.generic):  # numpy scalar
        return sanitize(obj.item(), key, _safe=safe)
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist(), key, _safe=safe)
    return hash_text(repr(obj))


def _dict_key(k: Any) -> str:
    """Dict keys are emitted verbatim only when they look like identifiers."""
    k = str(k)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]{0,63}", k) and scrub(k) == k:
        return k
    return hash_text(k)


def _safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("._")
    return s[:120] or "case"


def git_sha() -> str:
    env = os.environ.get("RADINTAKE_GIT_SHA") or os.environ.get("GITHUB_SHA")
    if env:
        return env
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parents[3], check=False)
        sha = out.stdout.strip()
        return sha if re.fullmatch(r"[0-9a-f]{40}", sha) else "unknown"
    except Exception:  # noqa: BLE001 - git missing is fine
        return "unknown"


def tool_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "radintake": __version__}
    for d in _TOOL_DISTS:
        try:
            out[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            continue
    return out


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


class CaseTrace:
    """Event stream for one case (one input file). Thread-safe, append-only."""

    def __init__(self, writer: TraceWriter, case_id: str, input_sha256: str):
        self.writer = writer
        self.case_id = case_id
        self.input_sha256 = input_sha256
        self.path = writer.run_dir / f"{case_id}.jsonl"
        self._seq = 0
        self._t0 = time.perf_counter()
        self._lock = threading.Lock()
        self.status: str | None = None
        self.closed = False

    @property
    def trace_id(self) -> str:
        return f"{self.writer.run_id}/{self.case_id}"

    def event(self, kind: str, **data: Any) -> dict:
        with self._lock:
            rec = {
                "schema_version": TRACE_SCHEMA_VERSION,
                "run_id": self.writer.run_id,
                "case_id": self.case_id,
                "seq": self._seq,
                "ts": _now_iso(),
                "t_ms": round((time.perf_counter() - self._t0) * 1000.0, 3),
                "kind": kind,
                "data": sanitize(data),
            }
            self._seq += 1
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")
            return rec

    # convenience wrappers --------------------------------------------------------------
    def step(self, stage: str, status: str, duration_ms: float, **data: Any) -> dict:
        return self.event("step", stage=stage, status=status, duration_ms=duration_ms, **data)

    def tool_call(self, tool: str, args: dict | None, output: Any, duration_ms: float,
                  tool_version: str | None = None) -> dict:
        return self.event("tool_call", tool=tool, tool_version=tool_version, args=args or {},
                          output=output, duration_ms=duration_ms)

    def model_call(self, model_id: str, prompt_id: str, prompt_hash: str, duration_ms: float,
                   **data: Any) -> dict:
        self.writer.register_model(model_id)
        self.writer.register_prompt(prompt_id, prompt_hash)
        return self.event("model_call", model_id=model_id, prompt_id=prompt_id,
                          prompt_hash=prompt_hash, duration_ms=duration_ms, **data)

    def close(self, status: str, **data: Any) -> None:
        if self.closed:
            return
        duration_ms = round((time.perf_counter() - self._t0) * 1000.0, 3)
        self.event("case_end", status=status, duration_ms=duration_ms, **data)
        self.status = status
        self.closed = True
        self.writer._case_closed(self, duration_ms)


class TraceWriter:
    """One per run. Creates ``<trace_dir>/<run_id>/`` and keeps the manifest current."""

    def __init__(self, trace_dir: str | Path, run_id: str | None = None, *,
                 model_ids: dict[str, str] | None = None, offline: bool | None = None,
                 extra: dict[str, Any] | None = None):
        self.run_id = _safe_name(run_id or (dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
                                            + "-" + uuid.uuid4().hex[:8]))
        self.run_dir = Path(trace_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._case_ids: set[str] = set()
        self.manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at": _now_iso(),
            "git_sha": git_sha(),
            "offline": offline,
            "model_ids": dict(model_ids or {}),
            "models_called": [],
            "prompts": {},
            "tool_versions": tool_versions(),
            "extra": sanitize(extra or {}),
            "cases": [],
        }
        self._write_manifest()

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def start_case(self, case_id: str, input_sha256: str, **data: Any) -> CaseTrace:
        with self._lock:
            base = cid = _safe_name(case_id)
            n = 1
            while cid in self._case_ids:
                n += 1
                cid = f"{base}-{n}"
            self._case_ids.add(cid)
            self.manifest["cases"].append({"case_id": cid, "input_sha256": input_sha256,
                                           "file": f"{cid}.jsonl", "status": None, "duration_ms": None})
            self._write_manifest()
        ct = CaseTrace(self, cid, input_sha256)
        ct.event("case_start", input_sha256=input_sha256, **data)
        return ct

    def register_prompt(self, prompt_id: str, prompt_hash: str) -> None:
        with self._lock:
            if self.manifest["prompts"].get(prompt_id) != prompt_hash:
                self.manifest["prompts"][scrub(prompt_id)] = scrub(prompt_hash)
                self._write_manifest()

    def register_model(self, model_id: str) -> None:
        with self._lock:
            if model_id not in self.manifest["models_called"]:
                self.manifest["models_called"].append(scrub(model_id))
                self._write_manifest()

    def _case_closed(self, case: CaseTrace, duration_ms: float) -> None:
        with self._lock:
            for c in self.manifest["cases"]:
                if c["case_id"] == case.case_id:
                    c["status"] = case.status
                    c["duration_ms"] = duration_ms
            self._write_manifest()

    def finalize(self) -> Path:
        with self._lock:
            self.manifest["finished_at"] = _now_iso()
            self._write_manifest()
        return self.manifest_path

    def _write_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, ensure_ascii=True, indent=2, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, self.manifest_path)
