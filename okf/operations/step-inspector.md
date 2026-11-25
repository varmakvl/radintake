---
type: Playbook
title: Step Inspector
description: Local debug tool that dumps the input and output of every pipeline step (load, preprocess, OCR, PHI masking, VLM calls, extraction, verification, DICOM de-identification) for any file, with an index.html to view the steps side by side.
resource: src/radintake/inspect.py
tags: [debugging, inspection, ocr, phi, vlm, dicom, demo]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-25T08:12:27.009Z
sources:
  - src/radintake/inspect.py
  - src/radintake/orchestrator/planner.py
  - tests/unit/test_inspect.py
---

# Use

```
python -m radintake.inspect <file-or-folder> [--out runs/inspect] [--pipeline ocr_first|vlm_first|hybrid]
```

Open `runs/inspect/<input name>/index.html`. The de-identified DICOM in `99_output/` and the
input `.dcm` can be compared in Weasis; rendered pages (PNG/JPEG/TIFF) open in any image viewer,
PDFs in a PDF viewer. Inputs for the corpus are in `data/corpus/files/` (PDF/TIFF/PNG/JPEG) and
`data/dicom_gt/inputs/` (DICOM).

# How it works

The [Orchestrator](/subsystems/orchestrator.md)'s planner calls `_inspect()` after every successful
stage; it is a no-op unless `RADINTAKE_INSPECT_DIR` is set, and any error in it is logged and swallowed
so inspection can never change a case outcome. `dump_stage` writes per stage:

| Folder | Contents |
|---|---|
| `00_input` | pages as loaded (original pixel grid) |
| `20_preprocess` | processed pages, `decisions.json` (metric -> action), composed transforms |
| `30_phi` | OCR word boxes, PHI detections on the original page, `page_N_masked.png` = exactly what a VLM may receive |
| `30_dicom_deid` | before / after (frame 0) and every header element that changed |
| `40_extract` | `report.json`, field boxes on the original page (blue OCR, green hybrid, purple VLM), VLM calls + cached responses |
| `50_verify`, `55_injection` | grounding / review flags, injection flags |
| `99_output` | the final cli_output.v1 JSON and masked / de-identified artefacts |

# Findings it surfaced on first use (2026-11-24)

- A technologist sign-off ("C. Rodriguez, RT") left unmasked on a phone photo: residual PHI the
  metrics also show (technologist recall 0.45).
- DICOM_001: burned-in name / DOB removed but the "[M]" token between them survived, and the
  "Semi-Upright / Portable" markers were blanked although TCIA keeps them - the reason burned-in pixel
  precision is 0.41.

# Guardrails & Anti-Regression Checklist

- Output contains PHI and IU report text: only ever under gitignored `runs/`; never commit or share.
- Inspection must stay outside the scored path: `eval.run` never sets `RADINTAKE_INSPECT_DIR`.
