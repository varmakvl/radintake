---
type: Configuration
title: Configuration
description: Layered runtime configuration (code defaults < configs/*.yaml < --config YAML < environment) covering seed, budgets, thresholds, the extraction pipeline choice, the model registry and offline/replay switches.
resource: src/radintake/config.py
tags: [config, yaml, environment, offline, budgets, models]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-24T11:48:24.228Z
sources:
  - src/radintake/config.py
  - configs/default.yaml
  - configs/models.yaml
  - configs/phi.yaml
  - configs/dicom_profile.yaml
  - configs/degrade_tiers.yaml
---

# Precedence

`DEFAULTS` in `config.py` < `configs/default.yaml` + `configs/models.yaml` (repo root or
`RADINTAKE_CONFIG_DIR`) < `--config` YAML < environment.

# Environment switches

| Variable | Effect |
|---|---|
| `RADINTAKE_OFFLINE=1` | local models only; `api` models raise `OfflineViolation` (CI, integration tests) |
| `RADINTAKE_FORBID_MODEL_CALLS=1` | every model call raises; used for re-scoring from traces |
| `RADINTAKE_DATA_DIR` | case store, review queue, VLM cache |
| `RADINTAKE_TRACE_DIR` | trace output root |
| `OLLAMA_HOST` | local Ollama server for the VLM and answer LLM (default `http://127.0.0.1:11434`; Compose uses `host.docker.internal`) |
| `TESSERACT_CMD` | Tesseract binary when not on PATH |

# Key settings (`configs/default.yaml`)

`seed: 140`; budgets (see [Orchestrator](/subsystems/orchestrator.md)); `loader.pdf_dpi: 200`;
preprocess thresholds ([Preprocessing Agent](/subsystems/preprocessing-agent.md));
`extraction.pipeline: ocr_first` plus VLM/hybrid knobs ([Extraction Agent](/subsystems/extraction-agent.md));
verifier thresholds ([Verifier](/subsystems/verifier-agent.md)); `injection.extra_patterns`.
Model registry: [Guarded Model Client](/subsystems/model-client-guard.md). Other files:
`phi.yaml` (anchors/regex), `dicom_profile.yaml` (Basic Profile actions), `degrade_tiers.yaml`.

# Guardrails & Anti-Regression Checklist

- Model ids and prompt versions must be recorded in every run manifest; changing them is a versioned change.
- Thresholds are provisional until tuned on the dev set — record the tuning in the report.
