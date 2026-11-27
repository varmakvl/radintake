---
type: Subsystem
title: DICOM De-identification Agent
description: Applies PS3.15 Basic Profile header actions with consistent UID remapping and redacts burned-in pixel text only inside detected boxes, re-encoding losslessly so the output stays valid DICOM.
resource: src/radintake/agents/dicom_deid.py
tags: [dicom, de-identification, phi, burned-in-text, pydicom]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-27T05:45:55.769Z
sources:
  - src/radintake/agents/dicom_deid.py
  - src/radintake/tools/dicom_io.py
  - configs/dicom_profile.yaml
  - data_prep/dicom_gt.py
  - tests/unit/test_dicom_deid.py
  - src/radintake/eval/metrics.py
  - eval/results/dicom_redaction_v2.json
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

# Burned-in text v2 (2026-11-27, `dicom-deid-2`)

`find_text_boxes` no longer depends on Tesseract word boxes to find text. Candidates: bright pixels
(`threshold` x max) that a white top-hat (`tophat_px`) keeps, split into glyph-sized components
(`min_char_px` .. `max_char_frac` x rows) and grouped into lines (heights within 1.5x, gap <= 1.2 x
height). A line must sit on a flat field (ring mostly dark, or flat and >= `min_contrast` darker
than the text) - this is what removes OCR garbage on bright anatomy - and be verified by a confident
word (>= 2 chars, `verify_conf`) from the full-frame OCR pass or a line-crop OCR, or, for 1-3 glyph
tokens (`[M]`, single mammography letters), a stroke/aspect check plus a one-character read
(`short_conf`). The keep list (`burned_in.safe_text` in `configs/dicom_profile.yaml`) keeps a line
when every confident word is a positional/laterality marker; 1-glyph fragments below `verify_conf`
do not count. Redacted lines of one block are merged (`merge_gap`). When the VOI window flattens a
frame to one value (DICOM_002), text search uses a min-max display.

Measured on the 48 TCIA files: text-pixel recall 0.614 -> 0.927, pixel precision 0.544 -> 0.825,
box recall 0.60 -> 1.00 (`eval/results/dicom_redaction_v2.json`). `metrics.json`
`dicom.per_file` lists per file: recall, precision, residual text pixels, changed pixels outside GT
and leaked tag keywords (ids and numbers only). Known false positives: study labels next to
redacted initials on the same field (DICOM_002 "CTU", "20MIN"), small anatomy fragments.

# Guardrails & Anti-Regression Checklist

- Never redact on the display-normalised array; always on raw stored values.
- Pixels outside redaction boxes must be unchanged (integration test allows a 15 px margin).
- Keep UID remapping consistent within a run so study/series links survive.
