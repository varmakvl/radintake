---
type: Subsystem
title: DICOM De-identification Agent
description: Applies PS3.15 Basic Profile header actions with consistent UID remapping and redacts burned-in pixel text only inside detected boxes, re-encoding losslessly so the output stays valid DICOM.
resource: src/radintake/agents/dicom_deid.py
tags: [dicom, de-identification, phi, burned-in-text, pydicom]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-24T12:24:50.803Z
sources:
  - src/radintake/agents/dicom_deid.py
  - src/radintake/tools/dicom_io.py
  - configs/dicom_profile.yaml
  - data_prep/dicom_gt.py
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

# Fail-safes and burned-in policy (2026-11-24)

Evaluation against the TCIA curated twins found header leaks outside Table E.1-1, so after the table
the agent also removes: every DA/DT element not named in the profile (Safe Harbor: all dates are PHI),
overlay groups 60xx, keyword prefixes ClinicalTrial* / ScheduledProcedureStep* / Admitting* / Curve* /
Overlay* / DateOfLast*, device IDs (PlateID, DetectorID, GantryID...) and procedure code sequences.
Dev header PHI recall 0.87 -> 0.998 (residual: two pixel-layout tags the curator rewrote, now excluded
from GT). Burned-in text is redacted per text LINE (single letters such as "[M]" between a name and a
date included) unless every confident word is a safe radiographic marker (`burned_in.safe_text`: L/R,
PA/AP, Portable, Upright...), which TCIA also keeps. Dev: burned-in box recall 0.13 -> 0.60, text-pixel
recall 0.61, pixel precision 0.41 -> 0.54. Known misses: text saturated by display windowing
(DICOM_002) and vertical mammography labels (DICOM_024) that OCR cannot read.

# Guardrails & Anti-Regression Checklist

- Never redact on the display-normalised array; always on raw stored values.
- Pixels outside redaction boxes must be unchanged (integration test allows a 15 px margin).
- Keep UID remapping consistent within a run so study/series links survive.
