---
type: Subsystem
title: Batch CLI
description: radintake batch --in <folder> --out <folder> writes one schema-valid JSON per input file, never stops on a bad file and prints only a count summary.
resource: src/radintake/cli.py
tags: [cli, batch, hidden-test, schema]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-23T17:39:22Z
sources:
  - src/radintake/cli.py
  - schemas/cli_output.v1.json
  - tests/integration/test_batch_cli_e2e.py
---

# Contract

```
radintake batch --in <folder> --out <folder> [--config <yaml>]
```

- One `<input-file-name>.json` per regular file directly inside `--in` (no recursion), validated
  against `cli_output.v1` ([Output & Trace Schemas](/data/output-schemas.md)).
- A crash inside `process_file` becomes an `error` document; the batch continues.
- Exit 0 when the batch ran (even with per-file errors); 2 only for invalid arguments.
- Traces to `RADINTAKE_TRACE_DIR` (default `<out>/_traces`); logs are JSON on stderr with PHI
  scrubbed; stdout is a single JSON count summary with `run_id`.

This is the interface the assessors' **hidden test** uses on unseen pages and DICOM, so it must be
robust to unknown formats and degradations. Implementation: [Orchestrator](/subsystems/orchestrator.md).
