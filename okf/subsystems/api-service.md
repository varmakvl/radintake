---
type: Subsystem
title: API Service
description: FastAPI app exposing POST /cases, GET /cases/{id}, GET /review-queue, POST /query and GET /health over a SQLite case store with background processing and problem+json errors.
resource: src/radintake/api/app.py
tags: [api, fastapi, service, sqlite, deployment]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-25T16:29:57.260Z
sources:
  - src/radintake/api/app.py
  - src/radintake/api/routes.py
  - src/radintake/store/cases.py
  - tests/unit/test_api_basic.py
  - tests/integration/test_api_e2e.py
  - tests/integration/test_ui_e2e.py
  - src/radintake/api/ui/index.html
  - src/radintake/api/ui/app.js
  - src/radintake/api/ui/app.css
---

# Routes

| Route | Behaviour |
|---|---|
| `POST /cases` | upload a file (or bundle) -> `case_id`; processed in the background via `pipeline.process_file` |
| `GET /cases/{id}` | status (`pending|running|ok|needs_review|error`), result document, flags, `trace_id`; 404 if unknown |
| `GET /review-queue` | items from the [review queue](/subsystems/verifier-agent.md) |
| `POST /query` | question -> cited answer or abstention; **retrieval not built yet** ([Retrieval / QA Agent](/subsystems/retrieval-qa-agent.md)) |
| `GET /cases/{id}/pages/{n}.png` | the PHI-MASKED page image written by the PHI agent (`phi.masked_artifacts`); 404 if the case or page is unknown; never the original upload |
| `GET /ui/` | review web UI (static files in `api/ui/`, mounted by `create_app`) |
| `GET /health` | `{"status": "ok", ...}`; used by the Compose healthcheck |

`create_app()` reads config from the environment (`RADINTAKE_DATA_DIR`, `RADINTAKE_OFFLINE`, ...).
Errors are `application/problem+json`. Logs carry ids and counts only.

# Review UI (`/ui`)

Plain HTML/JS/CSS in `src/radintake/api/ui/` (package data in `pyproject.toml`), mounted at `/ui`. Views:
upload, case (page image with field / PHI / table / cited boxes), review queue, query. Boxes are in
original page pixels and are only scaled to the displayed image. The `/ui` responses carry
`Content-Security-Policy: default-src 'self' ...` (`UI_CSP` in `app.py`), so the page cannot call out; the
Playwright smoke test records every browser request and asserts all go to localhost. The same CSP blocks
`eval`, so Playwright tests must wait on selectors, not `wait_for_function`. The case->output-dir map for the
page route is a marker file `<data>/cases/_index/<case_id>`.

# Storage

`store/cases.py`: `<RADINTAKE_DATA_DIR>/cases.sqlite`, one row per upload; rows are inserted or
advanced, never deleted. Survives restarts.

# Decision: decoded data lives in SQLite, as JSON documents (2026-11-24)

The OCR/VLM output for each case is stored as its JSON document in a column of the case store, not
appended to a per-GUID JSON file: background workers write concurrently, a crash mid-append
corrupts a plain JSON file, and retrieval needs indexed rows anyway. SQLite stays a single local
file and is transactional. Planned: a `subject_key` column (indexed) so "all records for one
GUID" is one query, plus an export to `<subject_key>.jsonl` for a file view. If the GUID is per
patient it must be pseudonymous (keyed hash of the patient ID with a local secret, or a random UUID
mapped locally), never the MRN itself. Decoded PHI (name, DOB, IDs) stays in this local file only —
never in logs or traces. Decided 2026-11-24: the patient and the uploading user are the same person,
so there is one key per patient: `subject_key = HMAC-SHA256(local secret, normalised patient ID)`.
Not implemented yet.

Deployment: [Deployment](/operations/deployment.md).
