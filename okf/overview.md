---
type: Architecture
title: System Overview
description: radintake turns scans, photos, PDF/TIFF and DICOM into PHI-safe, source-cited structured data through a fixed pipeline with agentic branches.
resource: src/radintake/pipeline.py
tags: [architecture, pipeline, agents, phi, overview]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-26T16:36:54.984Z
sources:
  - PLAN.md
  - requirements.md
  - docs/agents_vs_fixed.md
  - src/radintake/pipeline.py
---

# Purpose

radintake is the take-home deliverable for a Senior AI Engineer (OCR / VLM) assignment: a small
multi-agent system that ingests messy radiology inputs and produces structured, PHI-safe,
source-cited data a clinician can query, packaged as a service with an evaluation harness and a CI
regression gate. Scope and priorities: [Delivery Plan & Status](/playbooks/delivery-plan-and-status.md).

# Shape: a fixed pipeline with agentic branches

A step is an **agent** only when it chooses between actions based on what it observes and that
choice changes the result; everything deterministic stays a **fixed step** (cheaper, reproducible,
testable). The table of choices is in `docs/agents_vs_fixed.md`.

```
input file
  -> router (fixed)                      orchestrator/router.py
  -> planner runs the stage plan          orchestrator/planner.py + stages.py
       documents:  load(10) -> preprocess(20) -> phi(30) -> extract(40) -> verify(50) -> injection(55)
       DICOM:      dicom_deid(30)
  -> one cli_output.v1 JSON per input     pipeline.py
  -> review queue entry when flagged      store/review_queue.py
  -> (planned) index for retrieval / QA   retrieval-qa-agent
```

- [Orchestrator](/subsystems/orchestrator.md) — routing, budgets, retries, timeouts, failure semantics.
- [Preprocessing Agent](/subsystems/preprocessing-agent.md) — rule policy over image metrics.
- [Document PHI Agent](/subsystems/phi-agent.md) and [DICOM De-identification Agent](/subsystems/dicom-deid-agent.md).
- [Extraction Agent](/subsystems/extraction-agent.md) — OCR-first (current default), VLM-first, hybrid.
- [Verifier, Injection Detector & Review Queue](/subsystems/verifier-agent.md).
- [Retrieval / QA Agent](/subsystems/retrieval-qa-agent.md) — not built yet.

Entry points: [Batch CLI](/subsystems/batch-cli.md) and [API Service](/subsystems/api-service.md)
both call `pipeline.process_file`.

# Trust boundaries

1. **PHI boundary.** Only the PHI agent stamps payloads `phi_masked=True`; the
   [Guarded Model Client](/subsystems/model-client-guard.md) refuses anything else, so no external
   model ever sees unmasked pixels or text.
2. **Offline boundary.** `RADINTAKE_OFFLINE=1` refuses every `api` model; CI and the integration
   suite run this way. See [Configuration](/operations/configuration.md).
3. **Data-as-data.** Report text is never an instruction: the injection detector flags it, the
   verifier never rewrites values, and the planned QA agent spotlights document text as untrusted.
4. **Licence boundary.** IU Open-i (CC BY-NC-ND) renders and raw downloads never enter git; the repo
   is public. See [Data Preparation](/data/data-preparation.md).

# Cross-cutting contracts

- Every output box is in ORIGINAL page pixels: [Coordinate Contract](/data/coordinate-contract.md).
- Output shapes: [Output & Trace Schemas](/data/output-schemas.md).
- Every run is replayable from PHI-free traces: [Tracing & PHI-safe Logging](/subsystems/tracing-and-logging.md).

# Guardrails & Anti-Regression Checklist

- Never add a model call that bypasses `ModelClient.call`.
- Never let a stage write raw PHI into a trace, log line or review-queue entry.
- Never commit anything under `data/`, renders, degraded pages or real `.dcm` files.
- Keep `tests/integration/` as the end-to-end contract; do not weaken its assertions to pass.
