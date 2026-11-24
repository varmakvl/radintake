---
type: Operations
title: CI & Test Strategy
description: Unit tests per module, an end-to-end integration suite on synthetic licence-free fixtures that is the product contract, a GitHub Actions workflow running both offline, and the sprint test playbooks.
resource: .github/workflows/ci.yml
tags: [ci, testing, integration-test, github-actions, playbooks]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-24T17:01:04.519Z
sources:
  - .github/workflows/ci.yml
  - tests/integration/fixtures.py
  - tests/integration/test_batch_cli_e2e.py
  - tests/integration/test_api_e2e.py
  - integ-test-playbook.md
  - regression-test-playbook.md
  - pyproject.toml
---

# Layers

- **Unit** (`tests/unit/`, ~160 tests): one file per module; 2 skip when Tesseract is missing.
- **Integration** (`tests/integration/`, marker `integration`): the end-to-end contract.
  `fixtures.py` builds a licence-free folder at test time — clean PNG, rotated phone-photo JPEG,
  two-page PDF and TIFF with an injected instruction, a DICOM with header PHI and burned-in text,
  junk files. `test_batch_cli_e2e.py` runs the real CLI as a subprocess; `test_api_e2e.py` runs the
  app in-process. Assertions cover schema validity, routing, field accuracy, provenance IoU,
  injection handling, PHI masking and leakage, trace replay, DICOM validity and pixel preservation,
  cited answers and abstention.
- **Eval gate**: not built ([Evaluation Harness](/operations/evaluation-harness.md)).

# CI

`.github/workflows/ci.yml`: Python 3.11, apt `tesseract-ocr` + DejaVu fonts, lockfile install,
`ruff`, unit, integration (`RADINTAKE_OFFLINE=1`), then `radintake.eval.gate --subset ci` once
`eval/baseline.json` exists. The assignment requires one deliberate failing-gate commit followed by its fix.

# Known local issue

Tesseract is installed at `C:\Program Files\Tesseract-OCR` but is not on PATH for this shell, so
Tesseract-dependent tests skip locally; set `TESSERACT_CMD` or add it to PATH.

# Guardrails & Anti-Regression Checklist

- Never weaken or skip integration assertions to go green; file a bug instead (`integ-test-playbook.md`).
- Fixtures must stay synthetic — no IU or TCIA content in tests.
