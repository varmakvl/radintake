---
type: Subsystem
title: SDK Public API
description: radintake.sdk.Pipeline runs the full pipeline from Python on a path, bytes, a PIL image, a numpy array or a page list and returns a typed Result that dumps to a cli_output.v1 document; write-free unless output_dir, trace_dir or data_dir is given.
resource: src/radintake/sdk.py
tags: [sdk, api, public, results, pydantic, write-free]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-26T16:40:00.000Z
sources:
  - src/radintake/sdk.py
  - src/radintake/__init__.py
  - docs/api-stability.md
  - src/radintake/results.py
  - src/radintake/config.py
  - src/radintake/pipeline.py
  - schemas/cli_output.v1.json
  - tests/unit/test_sdk_pipeline.py
  - tests/unit/test_sdk_public_api.py
  - tests/integration/test_sdk_e2e.py
---

# Contract

```python
from radintake.sdk import Pipeline, RadintakeConfig

pipe = Pipeline(RadintakeConfig(offline=True))      # or a Config from load_config(), or None
r = pipe.process("report.pdf")                       # path | bytes | PIL image | ndarray | [pages]
r.status, r.flags, r.report.impression.value         # typed access
r.model_dump(mode="json")                            # a valid cli_output.v1 document
r.masked_pages, r.dicom_bytes                        # PHI-masked artefacts in memory (never dumped)
```

- `Result` (`results.py`, Pydantic v2) mirrors `schemas/cli_output.v1.json` field for field plus
  the optional `tables` array (`tables.v1`). Schema-nullable keys (`error`, `report`, `phi`,
  `dicom`, `field.value`, `table.section`) are always dumped; other optional keys are dropped
  when `None`; unknown keys are kept (`extra="allow"`), so pipeline output round-trips unchanged.
- `Pipeline.process` wraps `pipeline.process_file`; for the same `Config` its document equals
  `process_file`'s except `trace_id` (checked on every integration fixture kind).
- A bad input file is a `status: "error"` result, never an exception; an unsupported `source`
  type raises `UnsupportedInputError` (a `TypeError`).
- `process_batch(sources, output_dir=, on_result=)` returns one `Result` per source, in order; an
  unsupported source or an unexpected crash becomes an `error` result (`pipeline crashed: <Type>`,
  trace id `<run_id>/crashed`) and is written to `output_dir` like any other, so a batch never
  stops. `aprocess` / `aprocess_batch` / `radintake.aprocess` run the sync path in a worker thread
  (`asyncio.to_thread`).

# Public surface

`radintake.__all__` is the stable surface (semver policy: `docs/api-stability.md`): `Pipeline`,
`process`, `aprocess`, `RadintakeConfig`, `BudgetConfig`, `ModelConfig`, `Result` and its
sub-models, `RadintakeError`, `UnsupportedInputError`, `ConfigError`, `__version__`. They are
resolved lazily by a module `__getattr__`, so `import radintake` imports no FastAPI, uvicorn or
fastembed. Every other module is internal. The [Batch CLI](/subsystems/batch-cli.md) and the
[API Service](/subsystems/api-service.md) are thin wrappers over `Pipeline`.

# Write-free default

With no `output_dir` / `trace_dir` / `data_dir`, everything a case writes (spooled non-path
input, masked page PNGs, de-identified DICOM, the per-case trace, the `needs_review` queue row)
goes to a private `tempfile.TemporaryDirectory` deleted before `process` returns.
`phi.masked_artifacts` and `dicom.output_path` keep their `cli_output.v1` meaning (names relative
to `output_dir`); the files only exist on disk when an `output_dir` was given.

- `output_dir` -> `<output_dir>/<input>.json` plus artefacts and `review_queue.jsonl` (as the CLI).
- `trace_dir` -> one trace run per `Pipeline` under `<trace_dir>/<run_id>/` (`pipe.run_id`).
- `data_dir` -> persistent review queue (`review_queue.sqlite`) and model caches.
- `RADINTAKE_TRACE_DIR` / `RADINTAKE_DATA_DIR` do **not** turn persistence on for the SDK; the
  caller opts in explicitly.

# Configuration

`RadintakeConfig` is a frozen Pydantic model (offline, forbid_model_calls, budgets, pdf_dpi, seed,
ocr_engine, log_level, models, `sections` passthrough for stage sections, data_dir, trace_dir).
`to_config()` builds the internal `Config` through `config.config_from_raw` (laid over `DEFAULTS`,
no YAML read). `RADINTAKE_OFFLINE` / `RADINTAKE_FORBID_MODEL_CALLS` can only tighten it; the
[model guard](/subsystems/model-client-guard.md) re-reads them at call time. Schemas are resolved
through `config.schema_path(name)` so packaging can move them into package data in one place.

# Guardrails

- Never change `cli_output.v1`; `Result` follows the schema, not the other way round.
- Keep `process_file` behaviour-identical: the CLI, the API and eval depend on it.
- `StageContext.cancelled` is never reset per stage (known constraint) - the SDK does not try to.

# Tests

- `tests/unit/test_sdk_public_api.py`: one test per public entry point; every `Result` dump is
  validated against `cli_output.v1`; write-free check with cwd, TMP/TEMP, `RADINTAKE_DATA_DIR` and
  `RADINTAKE_TRACE_DIR` pointed at empty sandboxes (fails if a trace or masked PNG is written by
  default); `import radintake` checked in a subprocess for FastAPI/uvicorn/fastembed; the model
  guard still refuses unmasked payloads and offline API models under an SDK-built config.
- `tests/integration/test_sdk_e2e.py`: SDK vs batch CLI parity (status, flags, input_type, report
  values) on every synthetic fixture, bytes vs path parity, CLI summary counts.
