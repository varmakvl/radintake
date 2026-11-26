# API stability policy

radintake follows [Semantic Versioning 2.0.0](https://semver.org/): `MAJOR.MINOR.PATCH`.

- **PATCH** - bug fixes only; no change to the public surface.
- **MINOR** - backwards-compatible additions (new public names, new optional parameters, new
  optional result fields that `cli_output.v1` already allows) and deprecations.
- **MAJOR** - removals or incompatible changes to anything public.

While the version is `0.x` the same rules apply between minor releases, so `0.y` users can pin
`radintake~=0.y.0` safely.

## What is public

Exactly the names in `radintake.__all__`, importable from the top-level package:

| Name | What |
|---|---|
| `Pipeline` | reusable entry point: `process`, `process_batch`, `aprocess`, `aprocess_batch`, `close`, `run_id`, `config` |
| `process`, `aprocess` | one-shot convenience functions |
| `RadintakeConfig`, `BudgetConfig`, `ModelConfig` | explicit, immutable configuration (YAML optional) |
| `Result`, `Report`, `ReportField`, `Provenance`, `Phi`, `Detection`, `Dicom`, `Table`, `TableCell` | typed results |
| `RadintakeError`, `UnsupportedInputError`, `ConfigError` | public exceptions (`UnsupportedInputError` and `ConfigError` are also `TypeError`s) |
| `__version__` | package version |

Public means: the name, its documented parameters (names, order of positional ones, defaults)
and the documented attributes of the objects it returns.

Also public, with the same guarantees:

- the **batch CLI** contract `radintake batch --in <folder> --out <folder> [--config <yaml>]`:
  one `<input>.json` per input, exit codes 0 / 2, a single JSON count summary on stdout;
- the **output schema** `schemas/cli_output.v1.json`. `Result.model_dump(mode="json")`, the CLI
  files and the API `result` are all `cli_output.v1` documents. `cli_output.v1` is never changed
  incompatibly: a breaking change ships as a new schema version (`cli_output.v2`) alongside the old
  one for at least one minor release;
- the HTTP routes of the API service (`POST /cases`, `GET /cases/{id}`, `GET /review-queue`,
  `POST /query`, `GET /health`) and their response shapes.

## What is internal

Every other module and name - `radintake.pipeline`, `radintake.config`, `radintake.agents.*`,
`radintake.orchestrator.*`, `radintake.tools.*`, `radintake.store.*`, `radintake.tracing.*`,
`radintake.api.*`, `radintake.eval.*`, and any name starting with `_` - is internal. It may change
or disappear in any release without a deprecation period. Module docstrings of `radintake.sdk`
and `radintake.results` note which of their names are re-exported publicly; nothing else in them
is covered by this policy. Trace ids, timings and log lines are never part of the contract.

## Deprecation

A public name or parameter is removed only after it has emitted a `DeprecationWarning`
(stating the replacement and the release that removes it) for **at least one minor release**,
and the deprecation is listed in the release notes. Removal then happens in the next major
release (or, while `0.x`, no earlier than the minor release after the warning first shipped).
