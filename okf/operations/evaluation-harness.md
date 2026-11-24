---
type: Operations
title: Evaluation Harness
description: Scoring against ground truth into PHI-free score records, re-scorable run records, offline re-scoring with models forbidden, a prompt/schema/config version lock, a CI regression gate on a licence-free synthetic subset, and the MeSH-labelled golden QA set.
resource: src/radintake/eval/metrics.py
tags: [evaluation, metrics, golden-set, regression-gate, replay, versioning]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-24T14:00:00Z
sources:
  - src/radintake/eval/__init__.py
  - src/radintake/eval/metrics.py
  - src/radintake/eval/run.py
  - src/radintake/eval/rescore.py
  - src/radintake/eval/gate.py
  - src/radintake/eval/versions.py
  - src/radintake/eval/golden_qa.py
  - src/radintake/eval/error_analysis.py
  - eval/results/error_analysis_dev.md
  - eval/results/attempts.md
  - tests/unit/test_error_analysis.py
  - eval/baseline.json
  - eval/VERSIONS.lock
  - eval/golden_qa.jsonl
  - tests/unit/test_eval_harness.py
  - .github/workflows/ci.yml
---

# Commands

```
python -m radintake.eval.run --split dev [--pipeline ocr_first|vlm_first|hybrid] [--sample-per-tier K] [--dicom]
python -m radintake.eval.run --split golden --final --dicom   # once; writes eval/GOLDEN_USED.json, refuses a 2nd run
python -m radintake.eval.rescore --traces runs/<run> --check --verify-gt
python -m radintake.eval.gate --subset ci [--update-baseline]
python -m radintake.eval.versions [--update]
python -m radintake.eval.golden_qa
```

# Run record (runs/<name>/, gitignored; the final one is copied to traces/final/ minus outputs)

`outputs/` (values + PHI, never committed) · `traces/` (PHI-free, from [Tracing](/subsystems/tracing-and-logging.md)) ·
`scores.jsonl` (PHI-free score records) · `timings.json` (per case ms + pages, GPU peak via nvidia-smi
sampling) · `run.json` (split, pipeline, sample, git sha, model ids, prompt hashes, tool versions) ·
`metrics.json`.

# Score records (metrics.py)

Kinds `field`, `phi_gt`, `phi_det`, `doc`, `dicom`. They hold ids, flags, numbers and sha256 hashes of
normalised values — never values — so committed runs leak neither PHI nor licensed IU text, yet
`rescore --verify-gt` can prove (by re-hashing local GT) which ground truth they were scored against.

Definitions: normalised = whitespace-collapsed + casefolded; **EM** on normalised text; **CER** =
Levenshtein / len(GT); **hallucinated** = non-empty prediction with `partial_ratio` < 80 against all
printed field text of the document (a one-character OCR misread scores ~85 and stays a CER error —
the first threshold of 90 mislabelled such misreads, caught by a unit test); **omitted** = GT present,
prediction empty; **PHI detected** = a detection covers >= 50% of the GT instance box on the same page,
precision = detections with >= 50% of their area on GT PHI; **injection changed behaviour** = impression
echoes the injected text, turns "normal" against a non-normal GT, or a section leaks the patient name.
DICOM: valid, header PHI recall (GT tag value no longer present), UID remapping, burned-in recall
(>= 50% of the GT box pixels changed) and pixel precision (changed pixels inside GT boxes / all changed).
Aggregates: per field overall and per tier, macro mean over the brief's six fields, handwritten fields,
PHI by type and tier with every residual case, review share, latency p50/p95 per page, GPU peak, cost
($0 — open-source local models).

# Re-scoring and versions

`rescore` sets `RADINTAKE_FORBID_MODEL_CALLS=1` before importing anything; `--check` fails if the
recomputed metrics differ from the stored ones. `eval/VERSIONS.lock` pins sha256 of `prompts/*.md`,
`schemas/*.json` (immutable: a change must be a new `_vN`/`.vN` file) and `configs/*.yaml` (may change,
but only with a lock update). The gate checks the lock before scoring.

# CI gate (gate.py)

The CI subset is synthetic and licence-free: the 2 integration-fixture reports x 3 tiers through the
real `data_prep.degrade` (incl. the injected report) + the fixture DICOM with header PHI and burned-in
text. Rules vs `eval/baseline.json`: report-field mean EM >= base-0.10, CER <= base+0.08,
hallucination / omission <= base+0.10, document PHI recall >= base-0.10, injected changed behaviour
<= base, DICOM valid / burned-in recall >= base, header PHI recall >= base-0.05. Baseline at creation
(Windows, Tesseract 5): mean EM 0.67, CER 0.24, PHI recall 0.83, DICOM all 1.0.

# Golden QA (eval/golden_qa.jsonl)

32 questions over the golden documents: 22 MeSH-labelled answerable (relevance = golden docs whose
IU `mesh_major` matches, 1–38 docs each), 4 injection probes (one per injected golden study), 6
unanswerable (verified at build time to have no golden match — the builder caught one mislabelled
candidate, "shoulder prosthesis", because a golden study is coded `shoulder`).

# Error analysis (error_analysis.py)

`python -m radintake.eval.error_analysis --run runs/<run>` ranks pages by mean field CER (omission = 1)
+ 0.25 per missed PHI instance, categorises each failure (omission, hallucination, handwriting, wrong
field, layout: bleed / truncated, OCR misread, PHI miss) and draws overlays (GT green, prediction blue,
VLM purple, missed PHI red; local only). Dev v3 (`eval/results/error_analysis_dev.md`): 8 of the 10
worst pages are strongly keystoned phone photos where OCR reads almost nothing - the page-quad detector
misses the paper edge, so dewarp never runs. Across all pages: PHI miss 290, OCR misread 143, omission 106,
hallucination 99, layout bleed 42. A "skip CLAHE after flat-field" fix helped the worst pages but hurt the
typical photo and was rejected (`eval/results/attempts.md`).
