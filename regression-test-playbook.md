# regression-test-playbook.md — once per sprint (used by `regression-test-runner`)

## Permissions

- `.venv/Scripts/python`
- `python`

## Setup

`.venv/Scripts/python -m pip install -q -e ".[dev]"`

## Test scenario

1. `.venv/Scripts/python -m pytest tests -q --timeout=1800` (unit + integration) — 0 failed.
2. If `eval/baseline.json` exists: `.venv/Scripts/python -m radintake.eval.gate --subset ci` exits 0.

## Teardown

Remove `out/` test outputs created by the scenario.
