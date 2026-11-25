---
type: Subsystem
title: PDF Text Layer & Table Reading
description: Digital PDFs are read from their own text layer (exact words and boxes, hidden-text guard, OCR for mixed pages); a tables stage detects ruled, rules-only and borderless tables, re-reads ruled cells on OCR pages, and emits tables.v1 while keeping table text out of section prose.
resource: src/radintake/agents/tables.py
tags: [pdf, text-layer, tables, ocr, layout, prompt-injection, accuracy]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-25T08:12:27.009Z
sources:
  - src/radintake/tools/pdf_text.py
  - src/radintake/tools/tables.py
  - src/radintake/agents/tables.py
  - src/radintake/tools/loader.py
  - src/radintake/orchestrator/stages.py
  - src/radintake/agents/extract_ocr.py
  - src/radintake/agents/extract.py
  - schemas/tables.v1.json
  - data_prep/tables_set.py
  - src/radintake/eval/tables_eval.py
  - eval/results/tables_eval.json
  - tests/unit/test_pdf_text.py
  - tests/unit/test_tables.py
---

# Why (2026-11-25)

A born-digital PDF was rasterised and OCR'd, so exact text was turned into OCR errors, and only a
one-row "Exam Date / Modality" header line was parsed as a table. A synthetic CT report with a
3-row nodule table came out `status: ok` with the table garbled into Findings ("[enous [si Resove"):
silent data loss.

# PDF text layer (`tools/pdf_text.py`, loader)

`load_pdf` reads each page's pdfium text layer. Loose (advance-width) character boxes, so adjacent
glyphs touch and a gap means a space. They are mapped to the page raster (points x dpi/72, y
flipped), grouped into words and Tesseract-style line ids, and attached as `page.text_layer`
(ORIGINAL coordinates, conf 1.0). `Page.with_image` carries it through preprocessing, and
`page_words(page)` maps it onto the processed image, so every consumer still gets `OcrWord`s.

| Situation | Behaviour |
|---|---|
| ≥ `min_chars` (20) visible chars | text layer replaces OCR; preprocessing is skipped (`skip: pdf_text_layer`) |
| text-like ink outside the layer's words > `mixed_ink_threshold` (0.2; ruled lines removed first) | mixed page: OCR too, add OCR words that overlap no layer word |
| sparse layer, `/Rotate` page, broken layer | OCR as before |
| word with no ink under its box (white text), font < 4 pt, off the page | dropped, counted |

Hidden words are a prompt-injection channel. They are dropped, and the case is sent to review under
the v1 flag `prompt_injection_suspected`, with review reason code `hidden_text`. Config:
`loader.pdf_text_layer.{enabled,min_chars,min_font_pt,mixed_ink_threshold}`.

# Tables stage (`agents/tables.py`, order 25: after preprocess, before PHI)

1. Words once per page (text layer or Tesseract), cached as `ocr_words`; PHI, extraction and verify
   reuse them (PHI used to OCR).
2. `tools/tables.detect_tables`:
   - **Ruled detection.** Long horizontal rules come from morphology. Rules whose left and right
     ends agree (tolerance: max of 12 px, 0.6 x word height, 2% of page width) and lie within 16 x
     word height of each other form a stack. Vertical rules count only when they end inside the
     stack, so scanner scratches crossing the page are ignored.
   - **Splitting a stack.** A stack is split into runs of table-like bands:
     - with vertical rules: bands they cross (a header row OCR could not read still counts);
     - without vertical rules: bands whose words form at least 3 columns.
     So a patient key/value block, or prose between two same-width tables, never joins a table.
   - **Columns.** Taken from the vertical rules (`ruled`), or else from empty vertical strips in
     the word x-projection (`rules_only`, one row per text line).
   - **Borderless tables.** `aligned` finds runs of at least 2 consecutive lines with at least 3
     gap-separated segments. If their mean confidence is below 0.6 they are noise and are dropped.
3. On OCR pages each `ruled` cell is re-read on its own: crop inset 4 px, Otsu thresholding
   (inverted on shaded cells), upscale to 40 px, Tesseract `psm 7`. The re-read is kept when it is
   more confident. Re-read text replaces the cell's words in `ocr_words`, each table row with its
   own line id, so PHI masking and grounding see it too.
4. Output `tables` (tables.v1) with cell boxes in ORIGINAL coordinates, and `table_objs` (processed
   coordinates) for extraction.

# Extraction

`OcrFirstExtractor.map_fields(lines, tables)`:
- **Columns as fields.** Header-labelled columns of detected tables fill `exam_date`, `modality` and
  `accession_number` (row 0 = labels, row 1 = values) before the line heuristics run.
- **Accession only in detected tables.** `DETECTED_TABLE_LABELS` adds "Accession" for detected
  tables only. In the line-based fallback an "Accession No.: ..." header line itself looks like a
  table header; that happened, and accession came out as "Exam".
- **Table text stays out of sections.** Tokens inside a table are left out of the section prose,
  and the table records the section it sits in (`section`). All three pipelines (OCR-first, hybrid,
  VLM-first) get this through `extract_ocr.run`.
- **Weak tables go to review.** A table inside a section with confidence below
  `tables.min_section_table_conf` (0.6) adds `low_confidence` and review reason code
  `table_unreadable`.

# Contract

`cli_output.v1.json` is unchanged. It is locked in `eval/VERSIONS.lock`, and the brief names it as
the CLI contract. `tables` is an optional extra top-level key (v1 allows extra keys), described by
`schemas/tables.v1.json`. New conditions use existing v1 flags plus review-reason codes, so a v1
validator still accepts every output. See [Output & Trace Schemas](/data/output-schemas.md).

# Evidence

Synthetic table set (`python -m data_prep.tables_set`; 40 docs × digital / clean-scan / poor-scan
PDFs; grid, rules-only, borderless and shaded tables; 5 docs with hidden text):
`python -m radintake.eval.tables_eval` -> `eval/results/tables_eval.json` (aggregates only).
Results table: [Evaluation Harness](/operations/evaluation-harness.md).

# Limits (v1)

- **Cell structure:** no merged or spanning cells, nested tables or rotated tables.
- **Page-spanning tables:** a table that continues onto the next page is read as two tables.
- **Borderless tables on poor scans:** usually dropped, because OCR confidence there is below the
  0.6 floor.
- **Scanner text layers:** an invisible OCR layer that a scanner placed over the image passes the
  ink check, so it is trusted as is.
- **Photos:** the ruled detector needs rules that are roughly horizontal. Photos are dewarped first,
  but strongly curved pages fail.

# Guardrails & Anti-Regression Checklist

- Text-layer words must stay in ORIGINAL coordinates and be mapped to the current image with
  `page.to_current`; never store processed-coordinate words on the page.
- Never log or trace cell text or text-layer words (cells hold PHI): counts, shapes and confidences only.
- Do not add new values to `cli_output.v1` flags; use review-reason codes or a new schema version.
- Keep the accession label out of the line-based `table_hits` fallback.
- Re-run `tables_eval` and the dev eval after changing a detector threshold: the dev corpus guards
  against false positives (patient blocks, scratches, letterhead rules), the table set against misses.
