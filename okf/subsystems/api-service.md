---
type: Subsystem
title: API Service
description: FastAPI app exposing POST /cases, GET /cases/{id}, GET /review-queue, POST /query and GET /health over a SQLite case store with background processing and problem+json errors.
resource: src/radintake/api/app.py
tags: [api, fastapi, service, sqlite, deployment]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-23T17:39:22Z
sources:
  - src/radintake/api/app.py
  - src/radintake/api/routes.py
  - src/radintake/store/cases.py
  - tests/unit/test_api_basic.py
  - tests/integration/test_api_e2e.py
---

# Routes

| Route | Behaviour |
|---|---|
| `POST /cases` | upload a file (or bundle) -> `case_id`; processed in the background via `pipeline.process_file` |
| `GET /cases/{id}` | status (`pending|running|ok|needs_review|error`), result document, flags, `trace_id`; 404 if unknown |
| `GET /review-queue` | items from the [review queue](/subsystems/verifier-agent.md) |
| `POST /query` | question -> cited answer or abstention; **retrieval not built yet** ([Retrieval / QA Agent](/subsystems/retrieval-qa-agent.md)) |
| `GET /health` | `{"status": "ok", ...}`; used by the Compose healthcheck |

`create_app()` reads config from the environment (`RADINTAKE_DATA_DIR`, `RADINTAKE_OFFLINE`, ...).
Errors are `application/problem+json`. Logs carry ids and counts only.

# Storage

`store/cases.py`: `<RADINTAKE_DATA_DIR>/cases.sqlite`, one row per upload; rows are inserted or
advanced, never deleted. Survives restarts.

Deployment: [Deployment](/operations/deployment.md).
