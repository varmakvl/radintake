# integ-test-playbook.md — integration test playbook (used by the fleet-sprint `integ-test-runner`)

The integration suite is the end-to-end contract of the product. It runs the REAL pipeline
(no mocks) on synthetic, licence-free fixtures generated at test time
(`tests/integration/fixtures.py`): clean PNG, rotated phone-photo JPEG, multi-page PDF and
TIFF with a prompt-injection, a DICOM with header PHI + burned-in text, and junk files.

Do NOT weaken, skip or delete assertions in `tests/integration/` to make them pass. If an
assertion is genuinely wrong, file a bug explaining why instead.

## Permissions

- `.venv/Scripts/python`
- `python`
- `docker compose`
- `curl`

## Setup

1. Deploy has already run (`deploy.md`). Confirm `.venv/Scripts/python -m radintake.cli --help` exits 0.
2. `set RADINTAKE_OFFLINE=1` / `export RADINTAKE_OFFLINE=1` — local models only, deterministic.

## Test scenario

1. Functional suite (real pipeline, batch CLI + in-process API):
   `.venv/Scripts/python -m pytest tests/integration -m integration -q --timeout=1800`
   Pass = exit code 0 with 0 failed / 0 errors.
2. Hidden-test simulation (unseen folder through the CLI):
   `.venv/Scripts/python -c "from tests.integration.fixtures import build_corpus; from pathlib import Path; build_corpus(Path('out/integ_unseen'))"`
   then `.venv/Scripts/python -m radintake.cli batch --in out/integ_unseen --out out/integ_unseen_out`
   Pass = exit 0 and one `.json` per input file in `out/integ_unseen_out`.
3. Container (once `docker-compose.yml` exists): `curl -fsS http://localhost:8000/health`, then
   `curl -fsS -F "file=@out/integ_unseen/report_clean.png" http://localhost:8000/cases` returns a `case_id`;
   poll `GET /cases/{id}` until `status` is `ok` or `needs_review`.
4. Evaluation gate (once `eval/` exists): `.venv/Scripts/python -m radintake.eval.gate --subset ci` exits 0.

Feature-to-test mapping for closing beads:
- extraction / preprocessing / orchestrator features -> `test_report_extraction_*`, `test_provenance_*`, `test_input_routing`
- PHI (documents) -> `test_phi_masked_page_artifacts`, `test_no_phi_in_logs_or_stdout`
- PHI (DICOM) -> `test_dicom_deidentified_and_valid`
- verifier / injection -> `test_prompt_injection_is_treated_as_data`, review-queue part of `test_api_e2e`
- tracing / replay -> `test_traces_written_and_replayable`
- retrieval / QA -> `test_api_e2e::test_upload_query_and_review_flow`
- batch CLI robustness -> `test_one_schema_valid_json_per_input`, `test_junk_inputs_*`

## Teardown

Always: delete `out/integ_unseen` and `out/integ_unseen_out`; if `docker-compose.yml` exists, `docker compose down`.
