---
type: Subsystem
title: Tracing & PHI-safe Logging
description: Per-run manifests and per-case JSONL traces with every string hashed unless explicitly safe, plus structured JSON logs that pass through one shared PHI scrubber.
resource: src/radintake/tracing/writer.py
tags: [tracing, replay, observability, logging, phi, versioning]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-24T17:01:04.519Z
sources:
  - src/radintake/tracing/writer.py
  - src/radintake/logging.py
  - schemas/trace.v1.json
  - tests/unit/test_core_contracts.py
---

# Trace layout

```
<trace_dir>/<run_id>/manifest.json     run manifest: git sha, tool versions, model ids, offline flag
<trace_dir>/<run_id>/<case_id>.jsonl   one event per line (schemas/trace.v1.json)
```

`trace_dir` = `RADINTAKE_TRACE_DIR`, else `<out>/_traces` for the batch CLI. Events cover input
hash, stage steps, tool/model calls (model id, prompt hash, payload/output hashes, timings, cost),
preprocessing decisions and per-field value hashes + scores.

# PHI policy

Every string in an event payload becomes `sha256:<hex>` unless its key is in `SAFE_KEYS` (ids,
stage/tool names, statuses, model ids, prompt hashes). Safe values still pass the scrubber.
Re-scoring therefore compares stored hashes and scores, never raw text.

# Logging

`logging.py`: structlog JSON lines on stderr. `scrub_processor` replaces any value registered via
`register_phi` (the [PHI vault](/subsystems/phi-agent.md) registers every value) with `[REDACTED]`;
the trace writer uses the same registry. Policy: log ids, hashes, counts and timings only.

# Gaps

Offline re-scoring (`radintake.eval.rescore`) and prompt/schema version locking (`PROMPTS.lock`)
are not built: [Evaluation Harness](/operations/evaluation-harness.md).

# Guardrails & Anti-Regression Checklist

- Adding a new event key? It is hashed by default; only add it to `SAFE_KEYS` if it can never hold PHI or report text.
- Integration test scans every file under `_traces` for synthetic PHI strings.
