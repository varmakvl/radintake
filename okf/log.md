# Change Log

Newest changes recorded chronologically.


## 2026-11-24
- Synchronized timestamp for 3 drifted concept(s): operations/configuration.md, subsystems/extraction-agent.md, subsystems/model-client-guard.md.
- Fixed the Ollama backend: pages are downscaled to 1100 px (longest side) with boxes scaled back, and `num_ctx` raised to 8192; full-resolution pages overflowed the 4,096-token context. First live measurements recorded in model-client-guard.
- Recorded decisions: open-source local models only (Qwen2.5-VL-3B and Qwen2.5-3B via Ollama, Florence-2 crop fallback; Anthropic backend removed from `tools/vlm_client.py`), no vector database for retrieval (SQLite FTS5 + embedding BLOBs, brute-force cosine), decoded data kept as JSON documents in SQLite with a planned pseudonymous `subject_key`. Updated model-client-guard, extraction-agent, configuration, deployment, retrieval-qa-agent, api-service and delivery-plan-and-status.
- Synchronized timestamp for 6 drifted concept(s): operations/configuration.md, operations/deployment.md, subsystems/extraction-agent.md, subsystems/model-client-guard.md, subsystems/retrieval-qa-agent.md, subsystems/tracing-and-logging.md.

## 2026-11-23
- Synchronized steering notice in `CLAUDE.md` to v1.8.2.
- Synchronized workspace frontier mapping in `index.md` (+0 new, -1 removed, 42 already listed; 42 undocumented files mapped).
- Synchronized workspace frontier mapping in `index.md` (+39 new, -0 removed, 5 already listed; 44 undocumented files mapped).
- Bootstrapped the radintake OKF v0.2 bundle (nested layout: subsystems, data, operations, playbooks) from the code on `sprint/radintake-v1` at commit `0e903a4`, `PLAN.md`, `requirements.md` and the beads epic `rad-beb`. Concepts for not-yet-built parts (retrieval/QA, split and golden set, evaluation harness) are marked `status: draft`.
- Added the OKF steering section to `CLAUDE.md`.
