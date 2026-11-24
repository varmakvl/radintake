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
  at: 2026-11-24T00:00:00Z
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
| rad-beb.1 | Data acquisition + audit | IU download/parser written but uncommitted; TCIA + audit not started | [Data Preparation](/data/data-preparation.md) |
| rad-beb.2 | Rendering, handwriting, degradation, adversarial, GT | degradation library done; rendering not started | [Data Preparation](/data/data-preparation.md) |
| rad-beb.3 | Dedup, split, DICOM GT | not started | [Dataset Split & Golden Set](/data/dataset-split-and-golden-set.md) |
| rad-beb.4 | Orchestrator + preprocessing | built, tests closed | [Orchestrator](/subsystems/orchestrator.md) |
| rad-beb.5 | PHI (documents + DICOM) | built | [Document PHI Agent](/subsystems/phi-agent.md), [DICOM De-id](/subsystems/dicom-deid-agent.md) |
| rad-beb.6 | Extraction (OCR / VLM / hybrid) | all three built; evidence-based comparison pending | [Extraction Agent](/subsystems/extraction-agent.md) |
| rad-beb.7 | Verifier + review queue | built | [Verifier](/subsystems/verifier-agent.md) |
| rad-beb.8 | Tracing, eval harness, CI gate | trace writer built; eval harness not started | [Evaluation Harness](/operations/evaluation-harness.md) |
| rad-beb.9 | Retrieval / QA | not started | [Retrieval / QA Agent](/subsystems/retrieval-qa-agent.md) |
| rad-beb.10 | Deployment | API, Compose, lockfile built; not smoke-tested here | [Deployment](/operations/deployment.md) |
| rad-beb.11 | Final run, report, README | not started (README partly updated) | — |
| rad-beb.12 | LoRA bonus | not planned in detail | — |

Verified on 2026-11-23: 160 unit tests pass, 2 skipped (Tesseract not on PATH). The integration
suite was not run in this review.

# The key gap

The sprint built the runtime agents first. Priority 1 (the rendered dataset, split and ground
truth) is mostly missing, and without it there is no dev set to choose the extraction pipeline on
evidence, no golden numbers, and no report tables. Data generation and the evaluation harness are
the critical path.

# Open risks and decisions

- TCIA Pseudo-PHI-DICOM download may need the NBIA Data Retriever (manual step).
- Decided 2026-11-24: open-source local models only (Qwen2.5-VL-3B via Ollama, Florence-2 fallback, Qwen2.5-3B for answers); no vector database; decoded data in SQLite. Deadline 2026-12-02; LoRA bonus dropped.
- TCIA DICOM download is being done by the user via NBIA Data Retriever into `data/tcia/`; IU reports are fetched by script (the Open-i URL works again).
- Extraction pipeline choice is provisional (`ocr_first`).
- Tesseract PATH on the dev machine.
- Uncommitted `data_prep/` files and a modified `.beads/issues.jsonl` sit in the working tree.
- The fleet sprint is stopped; the fleet server and supervisor are not set to auto-start.
