---
type: Playbook
title: Delivery Plan & Status
description: Priority order of the take-home, mapping of features to beads, what is built and verified, what remains, and open risks and decisions as of the sprint stop on 2026-11-23.
resource: PLAN.md
tags: [plan, status, beads, sprint, risks, roadmap]
status: stable
stale_after: 2026-12-07T00:00:00Z
generated:
  by: Claude/opus-5.5
  at: 2026-11-27T05:36:18.985Z
sources:
  - PLAN.md
  - requirements.md
  - .beads/issues.jsonl
---

# Priority order (from the assignment)

1. Messy data and answer key. 2. Extraction, verification, PHI. 3. Harness and evaluation.
4. Chat with citations. 5. Deployment. 6. Bonus (LoRA). Incomplete submissions are judged on how
well the higher priorities are done.

# Status by feature (beads epic `rad-beb`, branch `sprint/radintake-v1`)

| Bead | Feature | State | Concept |
|---|---|---|---|
| rad-beb.1 | Data acquisition + audit | done 2026-11-24 (IU scripted, TCIA via NBIA, audit.md) | [Data Preparation](/data/data-preparation.md) |
| rad-beb.2 | Rendering, handwriting, degradation, adversarial, GT | done: 330 docs / 412 pages, 42% handwritten, 15 injected, 3 tiers, 4 formats | [Data Preparation](/data/data-preparation.md) |
| rad-beb.3 | Dedup, split, DICOM GT | done: golden 112 docs / 145 pages, 48 DICOM (6 burned-in) | [Dataset Split & Golden Set](/data/dataset-split-and-golden-set.md) |
| rad-beb.4 | Orchestrator + preprocessing | built, tests closed | [Orchestrator](/subsystems/orchestrator.md) |
| rad-beb.5 | PHI (documents + DICOM) | built | [Document PHI Agent](/subsystems/phi-agent.md), [DICOM De-id](/subsystems/dicom-deid-agent.md) |
| rad-beb.6 | Extraction (OCR / VLM / hybrid) | all three compared on dev (18 docs): OCR-first 0.642 / hybrid 0.651 / VLM-first 0.421 EM; OCR-first default | [Extraction Agent](/subsystems/extraction-agent.md) |
| rad-beb.7 | Verifier + review queue | built | [Verifier](/subsystems/verifier-agent.md) |
| rad-beb.8 | Tracing, eval harness, CI gate | done: metrics, rescore, version lock, CI gate (red `f5d4033`, green `19c9d6e`), error analysis on dev and golden | [Evaluation Harness](/operations/evaluation-harness.md) |
| rad-beb.9 | Retrieval / QA | built 2026-11-24; dev correctness 0.77, abstention 0.97, citations 1.0 | [Retrieval / QA Agent](/subsystems/retrieval-qa-agent.md) |
| rad-beb.10 | Deployment | done: Compose smoke test passed 2026-11-24 | [Deployment](/operations/deployment.md) |
| rad-beb.11 | Final run, report, README | done 2026-11-24: golden run once (`traces/final/`), 9-page PDF via `docs/report/build_report.py`, README | [Evaluation Harness](/operations/evaluation-harness.md) |
| rad-beb.12 | LoRA bonus | not planned in detail | — |

Dev numbers (OCR-first, 218 docs, 2026-11-24 after fixes): report-field mean EM 0.66 (clean 0.90, poor
0.57, photo 0.50), CER 0.17, hallucination 0.09, omission 0.08; document PHI recall 0.81 / precision
0.80; DICOM valid 1.0, header PHI recall 0.998, burned-in recall 0.60 (text-pixel 0.61, precision 0.54);
QA recall@5 0.58, answer correctness 0.77, correct abstention 0.97, citation accuracy 1.0; p50 2.6 s/page.
All 21 integration tests pass locally (hybrid and BM25-only).

Golden numbers (single run `20261124T170304Z-a5a6c448`, OCR-first, 112 docs / 145 pages): mean EM 0.672
(clean 0.88, poor 0.58, photo 0.55), CER 0.17; document PHI recall 0.83 / precision 0.81 (149 of 883
missed); DICOM valid 1.0, header PHI recall 1.0, burned-in text-pixel recall 0.61; QA recall@5 0.55,
answer correctness 0.72, correct abstention 0.97, citations 1.0; injection 1 of 4 flagged, 0 changed
behaviour; 60% of documents to review; p50 2.3 s / p95 3.5 s per page on CPU; $0 per page.
Golden lands within 2 points of dev, so dev tuning did not overfit. The DICOM set (48 files) is shared
by dev and golden and is not held out.

Remaining (optional) before 2026-12-02: close leftover beads; the LoRA bonus stays dropped.

# The key gap (closed)

The sprint built the runtime agents first; Priority 1 data, the evaluation harness, the pipeline
comparison and the golden run followed on 2026-11-24. The weakest area is now phone photos whose
page edge the quad detector misses (OCR reads almost nothing) and handwritten clinician names (PHI).

# Open risks and decisions

- Decided 2026-11-24: open-source local models only (Qwen2.5-VL-3B via Ollama, Florence-2 fallback, Qwen2.5-3B for answers); no vector database; decoded data in SQLite. Deadline 2026-12-02; LoRA bonus dropped.
- Extraction pipeline: `ocr_first` chosen on dev evidence; hybrid opt-in.
- Tesseract PATH on the dev machine.
- The fleet sprint is stopped; the fleet server and supervisor are not set to auto-start.
