# Change Log

Newest changes recorded chronologically.


## 2026-11-24
- Synchronized timestamp for 8 drifted concept(s): data/data-preparation.md, data/dataset-split-and-golden-set.md, operations/ci-and-test-strategy.md, operations/configuration.md, operations/deployment.md, subsystems/api-service.md, subsystems/dicom-deid-agent.md, subsystems/phi-agent.md.
- PHI misses: credential-anchored sign-off names, DICOM fail-safes (all undeclared DA/DT, overlay groups 60xx, clinical-trial / scheduled-step / device items), burned-in redaction by text line with a safe-marker allowlist, and burned-in scoring on exact text-pixel masks from the curated twins.
- Retrieval / QA built: `radintake.retrieval` (FTS5 BM25 + fastembed MiniLM, RRF) and `agents/qa.py` (extractive, verified citations, negation, abstention); `/query` wired; QA scoring (`score_qa`, `eval.run --qa / --qa-only`) and `eval/dev_qa.jsonl`. Dev: answer correctness 0.41 -> 0.77, abstention 0.97, citations 1.0. All 21 integration tests pass locally.
- Synchronized timestamp for 1 drifted concept(s): subsystems/orchestrator.md.
- Added the step inspector (`python -m radintake.inspect`, planner hook `_inspect`). Full dev re-run after the extraction fixes: mean EM 0.47 -> 0.66 (photos 0.18 -> 0.50), PHI recall 0.63 -> 0.81, precision 0.61 -> 0.80, injected pages flagged 5 -> 8 of 11.
- Extraction fixes driven by the first dev run: header column-gap / label stops, section stop markers for technologist notes, exam-date fallback, credential-name PHI rule + referring anchor, and preprocessing reordered to dewarp -> flat-field illumination -> orientation -> deskew. 30-doc dev sample: mean EM 0.44 -> 0.64, phone photos 0.07 -> 0.43, PHI recall 0.60 -> 0.83. CI baseline ratcheted.
- Synchronized timestamp for 6 drifted concept(s): data/coordinate-contract.md, operations/configuration.md, playbooks/delivery-plan-and-status.md, subsystems/extraction-agent.md, subsystems/phi-agent.md, subsystems/preprocessing-agent.md.
- Synchronized workspace frontier mapping in `index.md` (+2 new, -0 removed, 43 already listed; 45 undocumented files mapped).
- Synchronized workspace frontier mapping in `index.md` (+0 new, -1 removed, 43 already listed; 43 undocumented files mapped).
- Synchronized workspace frontier mapping in `index.md` (+2 new, -0 removed, 42 already listed; 44 undocumented files mapped).
- Evaluation harness: `radintake.eval` (metrics, run, rescore, gate, versions, golden_qa); PHI-free score records; `eval/VERSIONS.lock`, `eval/baseline.json`, `eval/golden_qa.jsonl` (32 questions). Hallucination threshold set to partial_ratio 80 after a unit test showed 90 counted one-character OCR misreads as hallucinations.
- Synchronized timestamp for 1 drifted concept(s): operations/ci-and-test-strategy.md.
- Priority 1 data work: IU download fixed (gzip check) and audited; 330 docs rendered with Playwright (seed-140 Faker identities, 42% handwritten pages, 15 injected docs, 82 multi-page); 3 tiers x 4 formats with homography-exact GT; MinHash dedup + StratifiedGroupKFold split (golden 112 docs / 145 pages, hash-locked); DICOM GT from 1,629 TCIA evaluation/curated pairs, 48 selected incl. all 6 burned-in. Data Preparation and Dataset Split concepts promoted to stable; subject_key decision (patient = uploader) recorded in api-service.
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
