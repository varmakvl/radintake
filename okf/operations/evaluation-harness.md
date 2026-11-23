---
type: Operations
title: Evaluation Harness
description: Planned metrics suite, golden QA set, offline re-scoring from traces, versioned prompts with a lock file, the CI regression gate against a stored baseline, and worst-page error analysis.
resource: PLAN.md
tags: [evaluation, metrics, golden-set, regression-gate, error-analysis, planned]
status: draft
generated:
  by: Claude/opus-5.5
  at: 2026-11-23T17:39:22Z
sources:
  - PLAN.md
  - requirements.md
  - .github/workflows/ci.yml
---

# Status

**Not built** (no `src/radintake/eval/`). Beads `rad-beb.8.1`–`rad-beb.8.8`. This is the largest
remaining block and it gates the evidence-based pipeline choice in the [Extraction Agent](/subsystems/extraction-agent.md).

# Planned components

- `radintake.eval.rescore --traces <dir> --offline`: recompute metrics from traces with
  `RADINTAKE_FORBID_MODEL_CALLS=1` ([Tracing](/subsystems/tracing-and-logging.md)).
- `PROMPTS.lock` with prompt/schema hashes and a test that fails on an unversioned change.
- Metrics on the golden set: per-field EM, CER, hallucination %, omission % (overall + per tier);
  document and DICOM PHI recall/precision with every residual case; burned-in recall/precision;
  DICOM validity; retrieval recall@5, citation accuracy, answer correctness, correct abstention;
  injected-page behaviour; % sent to review; latency p50/p95, GPU memory, cost per page.
- Golden QA: >= 25 questions with MeSH relevance labels, >= 5 unanswerable.
- `radintake.eval.gate --subset ci` vs `eval/baseline.json` thresholds; one intentional red commit.
- Error analysis: 10 worst golden pages, categorised, with overlays.

Inputs: [Dataset Split & Golden Set](/data/dataset-split-and-golden-set.md). Consumer: the final report
([Delivery Plan & Status](/playbooks/delivery-plan-and-status.md)).
