---
type: Subsystem
title: Guarded Model Client & VLM Client
description: Single choke point for every model call - refuses calls in replay mode, refuses API models offline, refuses payloads not masked by the PHI agent, then caches, times and traces the call; all models are open source and local (Ollama or in-process).
resource: src/radintake/tools/model_client.py
tags: [model-client, vlm, phi, offline, cache, cost, trust-boundary]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-25T08:12:27.009Z
sources:
  - src/radintake/tools/model_client.py
  - src/radintake/tools/vlm_client.py
  - configs/models.yaml
  - configs/default.yaml
  - tests/unit/test_extract_vlm_hybrid.py
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
parsing, crop payloads for the hybrid pipeline, and token usage in the trace. Backends, chosen per
model by `backend:` in the registry:

- `ollama` — `POST {OLLAMA_HOST}/api/chat` with the masked PNG, `format: json`, temperature 0,
  fixed seed. `ollama_available` only checks `/api/tags`; it never pulls a model.
- `transformers` — Florence-2-base in process, `<OCR>` crop transcription only, weights must
  already be in the HF cache (never downloaded at run time).

`select_vlm(task)`: the Ollama page VLM for any task, else Florence-2 for crops, else `None`
(callers fall back to OCR-first — this is what CI does).

# Decision: open-source, local models only (2026-11-24)

No paid API and no budget. Registry (`configs/models.yaml`), sized for a 4 GB RTX 3050:

| Name | Model | Runs on |
|---|---|---|
| `ocr` | Tesseract 5 | CPU |
| `vlm_page` | Qwen2.5-VL-3B (`qwen2.5vl:3b`, Q4 via Ollama, ~3.2 GB) | GPU, partly CPU-offloaded |
| `vlm_local` | Florence-2-base (0.23B) | GPU/CPU, crop fallback |
| `llm_answer` | Qwen2.5-3B (`qwen2.5:3b`, Ollama) | GPU |
| `embedder` | all-MiniLM-L6-v2 | CPU |

Cost per page is $0 in API terms; the report gives GPU-seconds and peak GPU memory instead. PHI
masking is still enforced before the VLM even though it is local: the guard does not depend on
where a model runs. The `api` kind and its offline refusal stay in the base class so a remote
model can never be added silently.

# Guardrails & Anti-Regression Checklist

- Every model call goes through this class — OCR, VLM, embedder, answer LLM.
- Never let the client trigger `ollama pull` or a Hugging Face download at run time.
- Local VLM calls can take tens of seconds: `extraction.vlm.timeout_s` is 180.
- On a failed Ollama attempt (empty / non-JSON reply, HTTP 500) the client unloads the model (`keep_alive: 0`)
  and retries once on a fresh load: the Qwen2.5-VL runner degrades after many image calls on a 4 GB GPU.
  Never send two image requests at once to the same local server during an evaluation run.
- Pages are downscaled to `max_image_side` (1100) before Ollama and any returned `bbox` is scaled
  back (`downscale` / `rescale_boxes`). Measured 2026-11-24 on the 4 GB GPU: a full 1700x2200 page
  is ~4k image tokens, fills the default 4,096 context, truncates the reply and takes ~64 s; at
  850x1100 it is ~1.2k tokens and ~4.5 s for one field. `num_ctx` is 8192. A full VLM-first page
  read (all sections + boxes, ~390 output tokens) took ~41 s, peak GPU ~2.2 GB; OCR-first and
  hybrid on a clean page took ~2 s (hybrid made no VLM call).
