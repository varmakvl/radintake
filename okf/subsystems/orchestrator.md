---
type: Subsystem
title: Orchestrator (router, planner, stage registry, pipeline)
description: Fixed input routing, a budgeted per-case state machine and a stage registry that assembles each input type's plan and writes exactly one output JSON per input.
resource: src/radintake/orchestrator/planner.py
tags: [orchestrator, router, planner, stages, budgets, retries, timeouts]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-24T17:01:04.519Z
sources:
  - src/radintake/orchestrator/router.py
  - src/radintake/orchestrator/planner.py
  - src/radintake/orchestrator/stages.py
  - src/radintake/pipeline.py
  - src/radintake/tools/loader.py
  - docs/agents_vs_fixed.md
  - tests/unit/test_router.py
  - tests/unit/test_planner.py
---

# Router (fixed step)

`orchestrator/router.py` classifies a file by magic bytes first, extension second: `DICM` at offset
128 -> `dicom`; `%PDF` -> `pdf`; TIFF magic -> `tiff`; PNG/JPEG -> `scan` or `photo` by a cheap
image heuristic (background unevenness, dark border, camera EXIF); otherwise a preamble-less DICOM
attempt, else `unknown`. Corrupt files are routed to an error, never raised. Deliberately not an
agent: deterministic, cheap, fuzz-tested.

# Stage registry

`orchestrator/stages.py`: each module registers its own stages with
`@register_stage(name, applies_to=..., order=..., failure_flag=..., critical=..., retries=...)`, so
parallel lanes never edit a shared list. `stages.discover()` imports `radintake.agents.*`.
Current plan:

| order | stage | applies to | module |
|---|---|---|---|
| 10 | `load` (critical) | documents | `orchestrator/stages.py` -> `tools/loader.py` |
| 20 | `preprocess` | documents | [Preprocessing Agent](/subsystems/preprocessing-agent.md) |
| 25 | `tables` | documents | [PDF Text Layer & Table Reading](/subsystems/tables-and-text-layer.md) (words once per page + tables) |
| 30 | `phi` | documents | [Document PHI Agent](/subsystems/phi-agent.md) |
| 30 | `dicom_deid` | DICOM | [DICOM De-identification Agent](/subsystems/dicom-deid-agent.md) |
| 40 | `extract` | documents | [Extraction Agent](/subsystems/extraction-agent.md) |
| 50 | `verify` | documents | [Verifier](/subsystems/verifier-agent.md) |
| 55 | `injection` | documents | [Verifier](/subsystems/verifier-agent.md) |

A stage gets a read-only `StageContext` and returns a dict merged into `ctx.data` only when it
succeeds in time.

# Planner (fixed state machine)

Budgets from [Configuration](/operations/configuration.md): `max_steps` 20, `stage_timeout_s` 300,
`retries` 1 with `backoff_s * 2**k`, `case_deadline_s` 900. Outcomes never raise:

- exception in a critical stage -> case `error`;
- exception in a non-critical stage -> `needs_review` + the stage's `failure_flag`, plan continues;
- timeout / step budget / deadline -> `timeout` flag; `needs_review` if partial results exist, else `error`.

# Pipeline

`pipeline.process_file(path, out_dir)` = route -> plan -> write `<out>/<input-name>.json`
validated against `cli_output.v1` ([Output & Trace Schemas](/data/output-schemas.md)). A document
that fails schema validation is replaced by an `error` document flagged `schema_error`. Shared by
the [Batch CLI](/subsystems/batch-cli.md) and the [API Service](/subsystems/api-service.md).

# Guardrails & Anti-Regression Checklist

- One input -> exactly one output JSON, whatever happens (junk and corrupt files included).
- Do not turn routing into a model call.
- New stages register themselves; never hard-code a stage list in the planner.
- A timed-out stage keeps running in an abandoned thread: stages must not mutate shared state.
