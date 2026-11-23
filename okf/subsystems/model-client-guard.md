---
type: Subsystem
title: Guarded Model Client & VLM Client
description: Single choke point for every model call - refuses calls in replay mode, refuses API models offline, refuses payloads not masked by the PHI agent, then caches, times, costs and traces the call.
resource: src/radintake/tools/model_client.py
tags: [model-client, vlm, phi, offline, cache, cost, trust-boundary]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-23T17:39:22Z
sources:
  - src/radintake/tools/model_client.py
  - src/radintake/tools/vlm_client.py
  - configs/models.yaml
  - prompts/extraction_v1.md
  - tests/unit/test_core_contracts.py
---

# ModelClient.call — checks in order

1. `RADINTAKE_FORBID_MODEL_CALLS=1` -> `ModelCallForbidden` (replay / re-score mode).
2. `RADINTAKE_OFFLINE=1` and an `api` model -> `OfflineViolation`.
3. Payload not stamped `phi_masked=True` -> `UnmaskedPayloadError`.
Then the backend call is timed and model id, prompt id + hash, payload/output hashes and timings are
recorded in the case trace ([Tracing & PHI-safe Logging](/subsystems/tracing-and-logging.md)).
Subclasses implement `_invoke` only.

# VlmClient

Adds a response cache keyed by `sha256(model id + payload hash + prompt hash)` under
`<data_dir>/cache/vlm` (replays are free; a cache hit still requires a masked payload), JSON
parsing, crop payloads for the hybrid pipeline, per-call cost from `usd_per_mtok_in/out`, and
`select_vlm` (API model, or local Florence-2-base only if weights are already cached — never
downloaded at run time).

Models (`configs/models.yaml`): `ocr` tesseract-5 (local), `vlm_local` Florence-2-base (local, fits
4 GB), `vlm_api` (API key from `ANTHROPIC_API_KEY` only), `embedder` MiniLM (local).

# Guardrails & Anti-Regression Checklist

- Every model call goes through this class — OCR, VLM, embedder, judge.
- The `vlm_api` id is currently `claude-sonnet-4-5`; refresh it to a current model before measured runs.
- Keys come from the environment only; never from config files committed to git.
