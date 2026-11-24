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
