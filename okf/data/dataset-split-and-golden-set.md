---
type: Pipeline
title: Dataset Split & Golden Set
description: Exact + MinHash-LSH near-duplicate groups, a report-level dev/golden split stratified by tier with groups never crossing and a hash-locked golden list, and DICOM ground truth derived from TCIA evaluation/curated pairs.
resource: data_prep/split.py
tags: [split, deduplication, golden-set, minhash, dicom-ground-truth]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-24T12:24:50.803Z
sources:
  - data_prep/split.py
  - data_prep/dicom_gt.py
  - data_prep/DICOM_GT.md
  - splits/SPLIT.md
  - splits/dev_ids.txt
  - splits/golden_ids.txt
  - splits/golden.lock
  - splits/dup_groups.json
  - splits/dicom_selection.csv
  - tests/unit/test_data_assets.py
---

# De-duplication

Findings + Impression normalised (lowercase; `XXXX`, digits, punctuation removed) -> exact pass ->
MinHash (128 perms, char 5-gram shingles) + LSH at Jaccard 0.8, every candidate re-checked with the
exact Jaccard -> union-find. Over the 330 corpus reports: 31 exact pairs, 90 verified near pairs,
**22 groups (62 reports), largest 11**. Manual check of the lowest-similarity merges (Jaccard
0.84–0.87): all were the same templated normal report — no false merges seen.

# Split

`StratifiedGroupKFold(3, shuffle, random_state=140)`, groups = dup groups, y = tier; golden = first
fold with >= 100 pages.

| | Docs | Pages | clean / poor / photo pages |
|---|---|---|---|
| Dev | 218 | 267 | 90 / 90 / 87 |
| **Golden** | **112** | **145** | 47 / 50 / 48 |

Golden holds 50 handwriting docs, 4 injected docs, 33 multi-page docs. `splits/golden.lock` is the
sha256 of `golden_ids.txt`; `tests/unit/test_data_assets.py` fails if the list changes or a dup group
crosses. All tuning uses dev; golden is scored once by the final run
([Evaluation Harness](/operations/evaluation-harness.md)).

# DICOM ground truth (data_prep/dicom_gt.py, data_prep/DICOM_GT.md)

The TCIA download has both halves: 21 evaluation patients (synthetic PHI) and 21 curated twins
(`Pseudo-PHI-001..021`); the UID crosswalk pairs 1,629 of 1,631 evaluation instances with their
curated copy. Header PHI = elements the curator removed/emptied/changed (remapped UIDs counted
separately); burned-in text = pixel diff of the pair -> connected components -> boxes. Only **6**
pairs have burned-in differences (CR 2, DX 2, MG 1, CT 1); all 6 are in the **48-instance selection**
(CR 4, CT 9, DX 10, MG 9, MR 8, PT 8). Inputs copied as `data/dicom_gt/inputs/DICOM_nnn.dcm` because
the original folder names contain synthetic patient names.

Known gaps: the GT encodes TCIA's policy (e.g. it keeps orientation markers like "Semi-Upright /
Portable / L" burned into a CR while removing the name/DOB line); text the curator did not clean is
invisible to a pixel diff; 2 instances have no curated twin.

Consumers: [Evaluation Harness](/operations/evaluation-harness.md), [DICOM De-identification Agent](/subsystems/dicom-deid-agent.md).
