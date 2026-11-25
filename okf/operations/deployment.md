---
type: Operations
title: Deployment
description: Docker image and Compose stack (CPU default, optional ocr-gpu profile) started with one command, health-checked, with pinned dependencies and the deploy runbook used by the sprint deployer.
resource: docker-compose.yml
tags: [deployment, docker, compose, healthcheck, lockfile]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-25T08:12:27.009Z
sources:
  - Dockerfile
  - .dockerignore
  - docker-compose.yml
  - requirements.lock
  - deploy.md
  - README.md
---

# Run

```
docker compose up -d --build --wait                    # API on http://localhost:8000
curl -fsS http://localhost:8000/health
docker compose --profile ocr-gpu up -d --build --wait  # optional NVIDIA variant on :8001
docker compose down
```

The image installs `tesseract-ocr`; data and traces live in the `radintake-data` volume (`/data`).
No API keys: the VLM and answer LLM run in Ollama on the host, reached through `OLLAMA_HOST`
(`host.docker.internal` via `extra_hosts: host-gateway`). Without Ollama the service still works
with OCR-first extraction. The image contains no data, renders or DICOM. Dependencies pinned in `requirements.lock` (`uv pip compile pyproject.toml --extra dev --universal`).

`deploy.md` is the runbook the fleet sprint's deployer follows (venv install + Compose + smoke test).
Service details: [API Service](/subsystems/api-service.md).

# Smoke test (verified 2026-11-24)

`docker compose up -d --build --wait`, then health ok, two synthetic reports (clean, rotated photo)
extracted 10 fields each, the synthetic DICOM completed, a corrupt PNG returned `status: error`
with a clear problem detail, `/query` answered with 4 verified citations (bm25+dense) and an
unanswerable question abstained. First build about 5 min (fastembed model baked into `/models`).

Gotcha: a host `uvicorn ... --port 8000` left running binds `127.0.0.1:8000`, while Docker binds
`::`/`0.0.0.0`; `curl localhost:8000` then reaches the stale host process (old code, 500 on
`/query`). Check `Get-NetTCPConnection -LocalPort 8000` before smoke-testing.
