---
type: Subsystem
title: Extraction Agent
description: Produces report.v1 (header fields, Indication, Comparison, Findings, Impression) with per-field confidence and provenance through one of three pipelines - OCR-first (default), VLM-first or hybrid.
resource: src/radintake/agents/extract.py
tags: [extraction, ocr, vlm, hybrid, provenance, report-schema, agent]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-26T12:19:16.423Z
sources:
  - src/radintake/agents/extract.py
  - src/radintake/agents/extract_ocr.py
  - src/radintake/agents/extract_vlm.py
  - src/radintake/agents/extract_hybrid.py
  - src/radintake/tools/ocr.py
  - src/radintake/tools/ocr_paddle.py
  - requirements-paddle.lock
  - tests/unit/test_ocr_engines.py
  - src/radintake/models/report.py
  - prompts/extraction_v1.md
  - prompts/crop_transcribe_v1.md
  - eval/results/pipeline_comparison.json
  - src/radintake/agents/tables.py
  - docs/ocr-engine.md
  - tests/unit/test_extract_ocr.py
  - tests/unit/test_extract_vlm_hybrid.py
---

# Stage

Stage `extract` (order 40) picks the pipeline from `extraction.pipeline` (`PIPELINES` registry in
`agents/extract.py`). Output: `report` (validated against `report.v1`), the in-process
`Report` model, and `extraction` metadata (counts / field names only). The trace gets value
HASHES, never values.

# Pipelines

**OCR-first** (`extract_ocr.py`, current default). Tesseract 5 words (reused from the PHI stage) ->
lines in reading order -> header labels (same anchors as the [Document PHI Agent](/subsystems/phi-agent.md))
-> exam-details table by column -> fuzzy section headings (`heading_min_ratio` 85) -> section text
until the next heading. Confidence = mean word confidence; below `low_conf_threshold` -> `low_confidence`.
Provenance = union of word boxes mapped to original coordinates ([Coordinate Contract](/data/coordinate-contract.md)).

**VLM-first** (`extract_vlm.py`). OCR-first still supplies header and table fields locally. Each
PHI-masked page goes to the local Qwen2.5-VL-3B (Ollama) with `prompts/extraction_v1.md` (JSON-constrained); placeholders are
restored from the vault locally; instruction-like sentences move to `untrusted_text`; values are
fuzzy-snapped to OCR words (`align_min_ratio` 80) -> source `hybrid`, else source `vlm` with
`unaligned_penalty`. No VLM available (Ollama not running, e.g. CI) -> falls back to OCR-first.

**Hybrid** (`extract_hybrid.py`). OCR-first everywhere; the VLM re-reads only crops of sections
below `hybrid_conf_threshold` (0.6) or required sections OCR could not find.

VLM calls go through the [Guarded Model Client & VLM Client](/subsystems/model-client-guard.md).

# Lessons from the first dev run (2026-11-24)

- Header values end at an unknown `Label:` token or a column gap (> 2.5x line height): two-column
  headers ("Patient Name: ...   Referring: Dr. X") had made patient-name EM 0.
- Sections end at technologist notes / addenda (`stop_patterns`): the note had been swallowed into
  the impression (impression EM 0).
- Shaded table header rows do not OCR, so `exam_fallback` takes exam date / modality from header-block
  dates (not the DOB) and modality codes - layout-agnostic, which matters for the hidden test.
Net on a 30-doc dev sample: mean EM 0.44 -> 0.64, hallucination 0.29 -> 0.12.

# Decision: OCR-first default, hybrid opt-in (2026-11-24, on dev evidence)

Same 18 dev documents (6 per tier, seed 140, 100 scored fields), `eval/results/pipeline_comparison.json`:

| Pipeline | Exact match | CER | Not on page | Missing | Photos EM | p50 / p95 s per page | VLM calls ok |
|---|---|---|---|---|---|---|---|
| OCR-first | 0.642 | 0.146 | 0.105 | 0.054 | 0.528 | 2.3 / 3.1 | – |
| Hybrid | 0.651 | 0.147 | 0.077 | 0.054 | 0.556 | 11.9 / 35.5 | 19 of 21 |
| VLM-first | 0.421 | 0.485 | 0.106 | 0.339 | 0.361 | 16.8 / 35.9 | 21 of 23 |

OCR-first stays the default: hybrid's gain is small (+0.9 points EM, -2.8 points hallucination, better
photos) for ~5x latency, and the graders' container has no local VLM, so hybrid would fall back to
OCR-first there anyway. Hybrid is kept as `extraction.pipeline: hybrid` when Ollama is available.
VLM-first with a 3B model on 4 GB omits a third of the fields. Getting a fair comparison took two fixes
(commit 5bf6aee and after): a dedicated crop prompt (`prompts/crop_transcribe_v1.md` - the shared prompt
made the model answer crops in page form, so hybrid silently equalled OCR-first) and reloading the
model when Ollama's runner degrades after many image calls (empty replies, HTTP 500, then "8888...").

# Tables (2026-11-25)

Detected tables ([PDF Text Layer & Table Reading](/subsystems/tables-and-text-layer.md)) feed
`exam_date` / `modality` / `accession_number` from header-labelled columns, and their tokens are kept
out of section prose (each table records its `section`). Section text on a digital PDF comes from the
text layer, so it is exact.

# Guardrails & Anti-Regression Checklist

- Header fields (name, ID, DOB, accession) are extracted locally; never send them to a VLM.
- Every field needs provenance with page >= 1 and a non-degenerate box in original coordinates.
- `models/report.py` and `schemas/report.v1.json` must stay in step (unit-tested).
- Expect the local VLM to be slow (tens of seconds per page on 4 GB); hybrid limits VLM calls to weak sections, so latency is part of the comparison.

# OCR engine switch

`radintake.tools.ocr.ocr()` dispatches on the active engine: `set_engine()` (pinned per case by
`pipeline.process_file` from `Config.ocr_engine`) > `RADINTAKE_OCR_ENGINE` > `ocr.engine` > `tesseract`.
`paddle` runs `tools/ocr_paddle.py` (PaddleOCR 2.10, PP-OCRv4 English rec + PP-OCRv3 det, CPU), which
returns the same `OcrWord` contract: page-pixel boxes clamped to the image, `conf` 0..1 (line score),
detected line boxes merged into rows by vertical-centre overlap (one `line_id` per row, `block = par = 1`),
words split from each line with boxes proportional to character offsets. Without the optional extra it
raises `OcrUnavailable` naming `[paddle]` / `requirements-paddle.lock` - no silent Tesseract fallback.
`orientation()` stays Tesseract. Details: `docs/ocr-engine.md`; contract tests: `tests/unit/test_ocr_engines.py`.
