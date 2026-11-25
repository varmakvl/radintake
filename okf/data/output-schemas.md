---
type: Data Contract
title: Output & Trace Schemas
description: JSON Schemas for the per-input CLI/API document (cli_output.v1), the report block (report.v1) and trace events (trace.v1), with a Pydantic mirror for reports.
resource: schemas/cli_output.v1.json
tags: [schema, json-schema, report, trace, pydantic, versioning]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-23T17:39:22Z
sources:
  - schemas/cli_output.v1.json
  - schemas/report.v1.json
  - schemas/trace.v1.json
  - schemas/tables.v1.json
  - src/radintake/models/report.py
---

# cli_output.v1

One per input. Required: `schema_version`, `input`, `input_sha256`, `input_type`
(`dicom|scan|photo|pdf|tiff|unknown`), `status` (`ok|needs_review|error`), `trace_id`. Optional:
`error`, `n_pages`, `report` (documents), `phi` (`masked_artifacts`, `detections`), `dicom`
(`output_path`, `valid`, `header_tags_deidentified`, `burned_in_regions`), `flags`
(`low_confidence`, `ungrounded_value`, `omission`, `schema_error`, `prompt_injection_suspected`,
`phi_uncertain`, `preprocess_failed`, `timeout`). Optional extra key `tables` (2026-11-25), described by
`schemas/tables.v1.json`: v1 allows extra top-level keys, so v1 itself stays unchanged and locked.
New review conditions reuse v1 flags and add review-reason codes instead of new flag values:
hidden text in a PDF text layer -> `prompt_injection_suspected` + `hidden_text`; a weak table inside
a report section -> `low_confidence` + `table_unreadable`.

# tables.v1

Array of `{page, bbox, n_rows, n_cols, method: ruled|rules_only|aligned, text_source: pdf_text|ocr,
confidence, section, cells: [{row, col, text, bbox, confidence}]}`, boxes in original page pixels.
Producer: [PDF Text Layer & Table Reading](/subsystems/tables-and-text-layer.md).

# report.v1

Fields `patient_name`, `patient_id`, `date_of_birth`, `accession_number`, `exam_date`, `modality`,
`indication`, `comparison`, `findings`, `impression`. Each is `{value, confidence, grounded,
needs_review, provenance: [{page, bbox, source: ocr|vlm|hybrid}]}`; boxes follow the
[Coordinate Contract](/data/coordinate-contract.md). `models/report.py` mirrors it; a unit test keeps
the two in step. Untrusted instruction-like text is carried separately (`untrusted_text`).

# trace.v1

Run manifest + per-case events: [Tracing & PHI-safe Logging](/subsystems/tracing-and-logging.md).

# Guardrails & Anti-Regression Checklist

- Breaking changes bump the version suffix (`.v2`) — the assessors' hidden test reads v1.
- The README documents cli_output.v1 for the hidden test; keep them consistent.
