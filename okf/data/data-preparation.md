---
type: Pipeline
title: Data Preparation
description: Scripted IU and TCIA acquisition, IU parsing and audit, rendering of 330 hospital-style report pages with synthetic seed-140 identities, handwriting and injected instructions, then three degradation tiers exported as PNG/JPEG/PDF/TIFF with box-exact ground truth.
resource: data_prep/generate.py
tags: [data, rendering, degradation, ground-truth, handwriting, adversarial, playwright, licence]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-24T17:01:04.519Z
sources:
  - data_prep/tables_set.py
  - data_prep/__init__.py
  - data_prep/download.py
  - data_prep/download_iu.py
  - data_prep/http.py
  - data_prep/manifest.py
  - data_prep/iu_parse.py
  - data_prep/audit.py
  - data_prep/audit.md
  - data_prep/fonts.py
  - data_prep/render.py
  - data_prep/degrade.py
  - data_prep/generate.py
  - configs/degrade_tiers.yaml
  - tests/unit/test_degrade.py
  - tests/unit/test_iu_parse.py
  - tests/unit/test_data_assets.py
  - pyproject.toml
---

# Commands (in order)

```
python -m data_prep.download iu --no-images   # 3,955 IU XML reports -> data/iu (sha256 manifest)
python -m data_prep.iu_parse                  # -> data/iu/reports.jsonl
python -m data_prep.audit                     # -> data_prep/audit.md (aggregates only, committed)
python -m data_prep.fonts                     # 4 OFL/Apache handwriting fonts -> data/fonts (sha256-pinned)
python -m data_prep.render                    # 330 docs -> data/generated (clean pages + clean GT)
python -m data_prep.generate                  # tiers, degradation, formats -> data/corpus
python -m data_prep.split                     # dedup + dev/golden -> splits/ (committed)
python -m data_prep.dicom_gt                  # TCIA pairs -> data/dicom_gt + splits/dicom_selection.csv
```
Install with `pip install -e ".[data]"` then `playwright install chromium`. IU X-ray PNGs are not
needed (pages are rendered from report text), so `--no-images` saves 1.3 GB. TCIA was downloaded
by hand with the NBIA Data Retriever into `data/tcia/` (both halves: evaluation + curated).

# Audit findings (data_prep/audit.md)

3,955 reports; Findings present 86.6%, Impression 99.1%; 35.2% coded MeSH `normal`; 24.1% share
their exact normalised Findings+Impression text with another report; `XXXX` (IU's own de-id token)
in up to 62% of sections — kept in rendered text and not PHI.

# Rendering (data_prep/render.py)

- 330 renderable reports drawn with numpy seed 140; identities from **Faker seeded 140** (name,
  MRN, DOB, accession, exam date, referring physician, radiologist, technologist).
- Jinja-style HTML per page, 3 hospital templates, patient block, **exam-details table**, sections,
  technologist note, signature. Multi-page (82 docs): page break before IMPRESSION with a repeated
  patient banner — so PHI appears on page 2 too.
- **Handwriting**: 40% of docs get Indication + technologist note + technologist name in one of 4
  handwriting fonts with per-word rotation/offset jitter -> **42% of pages** (brief: >= 30%).
- **Injected instructions**: 15 docs, 7 phrasings, placed in Findings, Impression or the
  technologist note. GT keeps `expected` (true content) separate from the printed text.
- Headless Chromium screenshots each page at 200 DPI -> 1700x2200 px (= pipeline `loader.pdf_dpi`);
  every value is a `data-f` element whose box comes from the live layout (per line / per handwritten
  word, unioned) — exact, not estimated.

# Degradation and export (data_prep/generate.py)

Tiers dealt round-robin after a seeded shuffle: 110 docs each of `clean_scan`, `poor_scan`,
`phone_photo`. One composed homography per page ([Coordinate Contract](/data/coordinate-contract.md)
idea): rotation incl. 90/180/270 (51 poor-scan and 37 photo pages turned), skew, perspective,
low-DPI scale, lighting, shadow, stamps, ruled lines, blur, noise, JPEG. Formats: PNG 116, JPEG 85,
PDF 82, TIFF 47 (multi-page docs alternate PDF/TIFF; PDFs embed pages at 200 DPI so the pipeline's
raster reproduces GT pixels). GT per doc in `data/corpus/gt/<doc>.json`: per page size, `H`, the
degrade op record, and per field instance the degraded quad + box, the clean box, `handwritten`, `phi`.
Visual QA overlays in `data/corpus/overlays/`.

# Licence and storage

IU Open-i is CC BY-NC-ND and the repo is public: raw downloads, renders, corpus files and GT live
under gitignored `data/` only. Committed: scripts, `audit.md` (aggregates), `splits/` (ids, hashes).

# Synthetic table set (2026-11-25)

`python -m data_prep.tables_set` -> `data/tables_set/` (gitignored): 40 fully synthetic reports
(Faker seed 140, template sentences, no IU text) with clinical tables (nodules, labs, prior studies,
CT dose; 2-8 body rows, 3-6 columns; grid / rules-only / borderless / shaded header; 8 docs with two
tables, 5 with hidden white or 1-pt text). Each in three variants: Chromium-printed digital PDF (text
layer) and the 200-DPI raster degraded to clean_scan / poor_scan, saved image-only. GT: field values,
table rows with their section, hidden-text flag. Scored by `radintake.eval.tables_eval`
([PDF Text Layer & Table Reading](/subsystems/tables-and-text-layer.md)).

Next: [Dataset Split & Golden Set](/data/dataset-split-and-golden-set.md).

# Guardrails & Anti-Regression Checklist

- Never commit anything under `data/`; never print report text into committed files.
- Every random choice is seeded (140 or a value derived from it) and recorded in GT / plan.json.
- System fonts (Arial, Times New Roman, Segoe UI) come from the rendering machine; renders are reproducible on Windows, not byte-identical elsewhere.
- Open-i can answer a download with an HTML page (HTTP 200): `download_iu` checks the gzip magic and re-fetches.
