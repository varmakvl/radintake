# Change Log

Newest changes recorded chronologically.

## 2026-11-26
- Synchronized timestamp for 2 drifted concept(s): playbooks/delivery-plan-and-status.md, subsystems/extraction-agent.md.
- Synchronized workspace frontier mapping in `index.md` (+1 new, -0 removed, 238 already listed; 239 undocumented files mapped).
- Synchronized timestamp for 3 drifted concept(s): operations/deployment.md, operations/evaluation-harness.md, subsystems/extraction-agent.md.
- Synchronized workspace frontier mapping in `index.md` (+1 new, -0 removed, 237 already listed; 238 undocumented files mapped).
- OCR engine trial on dev (same 60 ids): Tesseract stays the default for every input class (PaddleOCR EM 0.641 -> 0.537, 1.8x slower; higher PHI recall, lower precision). Result `eval/results/ocr_engine_comparison.json` bound to `subsystems/extraction-agent.md`; README results row, `docs/ocr-engine.md` and `eval/results/attempts.md` updated.
- Synchronized timestamp for 10 drifted concept(s): data/data-preparation.md, operations/ci-and-test-strategy.md, operations/configuration.md, operations/evaluation-harness.md, overview.md, playbooks/delivery-plan-and-status.md, subsystems/extraction-agent.md, subsystems/model-client-guard.md, subsystems/orchestrator.md, subsystems/preprocessing-agent.md.
- Synchronized workspace frontier mapping in `index.md` (+4 new, -0 removed, 233 already listed; 237 undocumented files mapped).
- OCR engine switch `ocr.engine` / `RADINTAKE_OCR_ENGINE` (tesseract default | paddle) with the optional PaddleOCR backend `src/radintake/tools/ocr_paddle.py` (extra `[paddle]`, pins in `requirements-paddle.lock`); `models.ocr_paddle` registry entry. Concepts `operations/configuration.md` and `subsystems/extraction-agent.md` updated; sources bound: `ocr_paddle.py`, `requirements-paddle.lock`, `tests/unit/test_ocr_engines.py`.
- Synchronized timestamp for 1 drifted concept(s): playbooks/delivery-plan-and-status.md.
- Synchronized timestamp for 1 drifted concept(s): playbooks/delivery-plan-and-status.md.
- Synchronized timestamp for 1 drifted concept(s): playbooks/delivery-plan-and-status.md.
- Phone-photo page-detection tests `tests/unit/test_page_detect.py` (synthetic low-contrast shadowed photos: line fallback fires, corners within 1% of diagonal, word-box round trip <= 1.5 px, identity on edge-less scans, 90/180 rotated photos upright); bound to `subsystems/preprocessing-agent.md`.
- Synchronized timestamp for 11 drifted concept(s): data/coordinate-contract.md, data/data-preparation.md, operations/ci-and-test-strategy.md, operations/configuration.md, operations/deployment.md, operations/evaluation-harness.md, playbooks/delivery-plan-and-status.md, subsystems/model-client-guard.md, subsystems/preprocessing-agent.md, subsystems/retrieval-qa-agent.md, subsystems/verifier-agent.md.
- Synchronized workspace frontier mapping in `index.md` (+12 new, -0 removed, 222 already listed; 234 undocumented files mapped).
- Phone-photo page detection: `image_ops.detect_page_quad` (contour quad kept only if every side is on a paper edge, else the Hough-line fallback `find_page_quad_lines`); trace fields `fallback`, `contour_rejected`, `score`; config keys `quad_fallback*`, `quad_contour_min_side_support`. Dev phone-photo EM 0.500 -> 0.614 (`eval/results/phone_photo_v2.json`). Concept `subsystems/preprocessing-agent.md` updated; source bound: `configs/default.yaml`.

## 2026-11-25
- Synchronized timestamp for 10 drifted concept(s): data/coordinate-contract.md, data/data-preparation.md, operations/ci-and-test-strategy.md, operations/configuration.md, operations/deployment.md, operations/evaluation-harness.md, playbooks/delivery-plan-and-status.md, subsystems/model-client-guard.md, subsystems/preprocessing-agent.md, subsystems/retrieval-qa-agent.md.
- Synchronized workspace frontier mapping in `index.md` (+2 new, -0 removed, 230 already listed; 232 undocumented files mapped).
- Synchronized timestamp for 10 drifted concept(s): data/data-preparation.md, operations/ci-and-test-strategy.md, operations/configuration.md, operations/deployment.md, operations/evaluation-harness.md, playbooks/delivery-plan-and-status.md, subsystems/api-service.md, subsystems/model-client-guard.md, subsystems/retrieval-qa-agent.md, subsystems/verifier-agent.md.
- Synchronized workspace frontier mapping in `index.md` (+9 new, -0 removed, 220 already listed; 229 undocumented files mapped).
- Synchronized timestamp for 4 drifted concept(s): data/data-preparation.md, data/dataset-split-and-golden-set.md, operations/evaluation-harness.md, playbooks/delivery-plan-and-status.md.
- Synchronized workspace frontier mapping in `index.md` (+15 new, -0 removed, 220 already listed; 235 undocumented files mapped).
- Golden v2 tests: `tests/unit/test_data_assets.py` (v2 lock, edited-id detection, disjointness from dev/golden v1,
  no shared dup group, v1 lock pinned) and `tests/unit/test_eval_harness.py` (`--split golden_v2` needs `--final`,
  second run refused, tampered lock refused, v1 guard still refuses, `--check-only`); guards patched to tmp paths.
- Synchronized timestamp for 3 drifted concept(s): data/dataset-split-and-golden-set.md, operations/evaluation-harness.md, playbooks/delivery-plan-and-status.md.
- Synchronized workspace frontier mapping in `index.md` (+8 new, -0 removed, 220 already listed; 228 undocumented files mapped).
- Golden v2 held-out set: new `data_prep/heldout.py` (seed 241; 112 docs / 140 pages from IU reports never used in v1;
  697 near-duplicates of v1 dropped, zero shared dup groups), `splits/golden_v2_ids.txt` + `golden_v2.lock`,
  and `eval.run --split golden_v2` (requires `--final`, own guard `eval/GOLDEN_V2_USED.json`, `--check-only`).
  Updated concept [Dataset Split & Golden Set](/data/dataset-split-and-golden-set.md).
- Synchronized timestamp for 1 drifted concept(s): operations/evaluation-harness.md.
- Synchronized timestamp for 8 drifted concept(s): operations/deployment.md, operations/evaluation-harness.md, operations/step-inspector.md, overview.md, playbooks/delivery-plan-and-status.md, subsystems/model-client-guard.md, subsystems/preprocessing-agent.md, subsystems/tables-and-text-layer.md.
- Synchronized workspace frontier mapping in `index.md` (+4 new, -0 removed, 216 already listed; 220 undocumented files mapped).
- PDF text layer and table reading: new concept [PDF Text Layer & Table Reading](/subsystems/tables-and-text-layer.md).
  - New pieces: `tools/pdf_text`, `tools/tables`, the `tables` stage (order 25), `schemas/tables.v1.json`,
    `data_prep.tables_set` and `eval.tables_eval`.
  - Digital PDFs: field EM 0.658 -> 1.000 and tables 0% -> 100% (all cells exact); hidden text flagged in 5 of 5.
  - Dev: exam date EM 0.50 -> 0.60, PHI recall 0.827 -> 0.848, p95 latency 14.2 -> 5.0 s.
  - Contract: `cli_output.v1` is unchanged; new conditions use v1 flags plus review-reason codes.
  - Updated concepts: orchestrator, coordinate contract, output schemas, extraction, PHI, configuration,
    data preparation and evaluation harness.

## 2026-11-24
- Synchronized timestamp for 2 drifted concept(s): operations/deployment.md, operations/evaluation-harness.md.
- Golden run done once (`traces/final/`), golden error analysis, PDF report builder (`docs/report/build_report.py`), README results, Compose smoke test verified, CI gate step now runs even when unit tests fail.
- Synchronized timestamp for 15 drifted concept(s): data/data-preparation.md, data/dataset-split-and-golden-set.md, operations/ci-and-test-strategy.md, operations/configuration.md, operations/deployment.md, operations/evaluation-harness.md, operations/step-inspector.md, overview.md, subsystems/batch-cli.md, subsystems/orchestrator.md, subsystems/phi-agent.md, subsystems/preprocessing-agent.md, subsystems/retrieval-qa-agent.md, subsystems/tracing-and-logging.md, subsystems/verifier-agent.md.
- Synchronized workspace frontier mapping in `index.md` (+1 new, -0 removed, 47 already listed; 48 undocumented files mapped).
- Synchronized timestamp for 1 drifted concept(s): playbooks/delivery-plan-and-status.md.
- Synchronized workspace frontier mapping in `index.md` (+1 new, -0 removed, 46 already listed; 47 undocumented files mapped).
- Error-analysis tool (`radintake.eval.error_analysis`); dev worst pages are keystoned photos where dewarp never runs. Tried-and-rejected log started in `eval/results/attempts.md`.
- Synchronized timestamp for 2 drifted concept(s): subsystems/extraction-agent.md, subsystems/model-client-guard.md.
- Synchronized workspace frontier mapping in `index.md` (+1 new, -0 removed, 45 already listed; 46 undocumented files mapped).
- Pipeline comparison on 18 dev docs: OCR-first 0.642 EM / 2.3 s, hybrid 0.651 / 11.9 s, VLM-first 0.421 / 16.8 s. OCR-first kept as default, hybrid opt-in. Fixes found on the way: dedicated crop prompt; reload the VLM when Ollama's runner degrades. Progress dashboard published as a private artifact.
- Synchronized timestamp for 3 drifted concept(s): data/dataset-split-and-golden-set.md, subsystems/dicom-deid-agent.md, subsystems/phi-agent.md.
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
- 2026-11-25: Documented the review UI at `/ui` and the PHI-masked page route `GET /cases/{id}/pages/{n}.png` in `subsystems/api-service.md` (sources bound: ui assets, Playwright test).
