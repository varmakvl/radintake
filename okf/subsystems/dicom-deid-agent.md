---
type: Subsystem
title: DICOM De-identification Agent
description: Applies PS3.15 Basic Profile header actions with consistent UID remapping and redacts burned-in pixel text only inside detected boxes, re-encoding losslessly so the output stays valid DICOM.
resource: src/radintake/agents/dicom_deid.py
tags: [dicom, de-identification, phi, burned-in-text, pydicom]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-23T17:39:22Z
sources:
  - src/radintake/agents/dicom_deid.py
  - src/radintake/tools/dicom_io.py
  - configs/dicom_profile.yaml
  - tests/unit/test_dicom_deid.py
---

# Header (fixed profile)

`configs/dicom_profile.yaml` gives PS3.15 Basic Application Confidentiality Profile actions per
keyword: remove (X), zero-length (Z), dummy (D), consistent UID remap (U), keep (K). Fail-safe rules
then empty every other PN element, remap every other non-registered UID and drop private tags unless
safe-listed; sequences are handled recursively. Sets `PatientIdentityRemoved=YES`,
`DeidentificationMethod` and the method code sequence (113100, plus 113101 Clean Pixel Data when
pixels were redacted).

# Burned-in text (agentic part)

Per frame: `dicom_io.to_display` windows the stored values to uint8 FOR OCR ONLY ->
`find_text_boxes` -> `redact_frames` blacks out the RAW stored values inside those boxes only.
`dicom_io.write_frames` re-encodes with the original transfer syntax when lossless and an encoder
exists (pixels outside boxes stay bit-identical), else Explicit VR Little Endian with updated file
meta. `dicom_io.validate` re-reads the result; `dicom.valid` is reported in the output.

Stage `dicom_deid` (order 30, `failure_flag: phi_uncertain`). Ground truth for evaluation comes
from the TCIA curated counterpart: [Dataset Split & Golden Set](/data/dataset-split-and-golden-set.md).

# Guardrails & Anti-Regression Checklist

- Never redact on the display-normalised array; always on raw stored values.
- Pixels outside redaction boxes must be unchanged (integration test allows a 15 px margin).
- Keep UID remapping consistent within a run so study/series links survive.
