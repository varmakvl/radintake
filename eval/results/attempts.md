# Tried and rejected / kept attempts (dev set evidence)

Each line: what was tried, the evidence, the decision. Numbers are report-field mean exact match (EM)
unless stated; samples are seeded dev subsets, never the golden set.

| Date | Attempt | Evidence | Decision |
|---|---|---|---|
| 2026-11-24 | Header value stops at a column gap / unknown `Label:` | full dev: patient-name EM 0.00 -> 0.69 | kept |
| 2026-11-24 | Section ends at technologist note / addendum | full dev: impression EM 0.01 -> 0.55 | kept |
| 2026-11-24 | Flat-field illumination + dewarp before orientation | full dev: phone-photo EM 0.18 -> 0.50 | kept |
| 2026-11-24 | Hybrid with the shared extraction prompt for crops | 17/17 crop re-reads failed (model answered in page form) | replaced by `crop_transcribe_v1` |
| 2026-11-24 | Pipeline choice: OCR-first vs hybrid vs VLM-first (18 docs) | 0.642 / 0.651 / 0.421 EM; 2.3 / 11.9 / 16.8 s per page | OCR-first default, hybrid opt-in |
| 2026-11-24 | Skip CLAHE after flat-field (from worst-10 error analysis) | worst 10: omission 0.74 -> 0.67; 18-doc sample: photo EM 0.53 -> 0.47, PHI recall 0.84 -> 0.80 | rejected (helps the worst pages, hurts the typical photo) |
| 2026-11-24 | Looser page-quad contrast threshold (0.3, 0.15) | finds fewer quads than the default 0.6 on 5 worst photos | rejected |
| 2026-11-25 | Tight pdfium glyph boxes for text-layer word splitting | "Measu rements", "0001 2345": narrow glyphs leave gaps read as spaces | replaced by loose (advance-width) boxes, gap > 0.15 x height |
| 2026-11-25 | "Accession" as a label in the line-based exam-table fallback | dev doc D0011: the "Accession No.: ..." header line was read as a table header, accession = next line | label used for detected tables only |
| 2026-11-25 | Re-read cell words keep the OCR line id of their row | accession = "Exam": OCR line ids around the table run into the patient block | each re-read table row gets its own line id |
| 2026-11-25 | Rules stack only when the gap is < 6 x word height | rules-only tables with > 4 body rows missed (table set: rules 0.38 cell EM) | 16 x word height, then split stacks at bands that are not table-like |
| 2026-11-25 | Split a rule stack at bands whose words do not form columns, ruled or not | patient key/value block (4 columns) joined the exam table | with vertical rules, only bands they cross count |
| 2026-11-25 | Require OCR words inside a ruled grid | Tesseract psm 6 returns no words for dense grids: 6-column dose tables never found on scans (shaded 0.48 cell EM) | keep word-less grids, read every cell with the per-cell re-read, drop grids still empty |
| 2026-11-25 | Spread re-read cell text evenly over the cell box | dev PHI precision 0.807 -> 0.787 (boxes wider than the text miss GT boxes) | keep Tesseract's own word boxes from the cell crop |
| 2026-11-25 | Table column fills accession before the header line | CI fixture CI002: borderless detector paired "Accession No.:" with the table header -> accession "Exam" | header line wins for accession; single-band ruled tables split by text line |
| 2026-11-26 | Line-based page detector only when the contour detector finds NO quad | dev detector scan: 6 of 87 photo pages have no contour quad, but 64 more have a WRONG one (a side not on any paper edge: the lighting gradient merges a bright desk corner into the page) | extended: also replace contour quads that fail the edge check |
| 2026-11-26 | Contour quad kept only if every side lies on a paper edge (support >= 0.5), else Hough-line quad (`find_page_quad_lines`) | full dev, 218 docs, same ids: phone-photo EM 0.500 -> 0.614, omission 16.0% -> 6.6%, PHI recall 0.763 -> 0.881; clean / poor scan unchanged (0.903 / 0.571); worst-10 omission 0.75 -> 0.25; 70 photo pages use the line quad, 0 scans dewarped ([phone_photo_v2.json](phone_photo_v2.json)) | kept |
| 2026-11-26 | OCR engine PaddleOCR PP-OCRv4 (`ocr.engine: paddle`) vs Tesseract 5, OCR-first, offline, CPU | dev, same 60 ids (20 per tier, seed 140): EM 0.641 -> 0.537 (clean 0.861 -> 0.792, poor 0.511 -> 0.402, photo 0.547 -> 0.413; by router input class scan -3.8, photo -13.5, pdf -13.6, tiff -7.6 points); CER 0.157 -> 0.107; PHI recall 0.889 -> 0.937, precision 0.775 -> 0.714; p50 per page 3.6 s -> 6.5 s; GPU 0 for both ([ocr_engine_comparison.json](ocr_engine_comparison.json)) | rejected as default and per input class: Tesseract everywhere; paddle stays an opt-in engine (no Docker build arg, default model_ids unchanged) |
