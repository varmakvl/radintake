---
type: Subsystem
title: Extraction Agent
description: Produces report.v1 (header fields, Indication, Comparison, Findings, Impression) with per-field confidence and provenance through one of three pipelines - OCR-first (default), VLM-first or hybrid.
resource: src/radintake/agents/extract.py
tags: [extraction, ocr, vlm, hybrid, provenance, report-schema, agent]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-24T10:31:35.389Z
sources:
  - src/radintake/agents/extract.py
  - src/radintake/agents/extract_ocr.py
  - src/radintake/agents/extract_vlm.py
  - src/radintake/agents/extract_hybrid.py
  - src/radintake/tools/ocr.py
  - src/radintake/models/report.py
  - prompts/extraction_v1.md
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

# Open decision

The assignment requires choosing the pipeline **on evidence** from a dev-set comparison (EM, CER,
hallucination, omission, latency, cost per tier). That comparison needs the rendered dev set and
the [Evaluation Harness](/operations/evaluation-harness.md), neither of which exists yet, so
`ocr_first` is a provisional default. Tracked in [Delivery Plan & Status](/playbooks/delivery-plan-and-status.md).

# Guardrails & Anti-Regression Checklist

- Header fields (name, ID, DOB, accession) are extracted locally; never send them to a VLM.
- Every field needs provenance with page >= 1 and a non-degenerate box in original coordinates.
- `models/report.py` and `schemas/report.v1.json` must stay in step (unit-tested).
- Expect the local VLM to be slow (tens of seconds per page on 4 GB); hybrid limits VLM calls to weak sections, so latency is part of the comparison.
