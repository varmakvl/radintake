---
type: Operations
title: Deployment
description: Docker image and Compose stack (CPU default, optional ocr-gpu profile) started with one command, health-checked, with pinned dependencies and the deploy runbook used by the sprint deployer.
resource: docker-compose.yml
tags: [deployment, docker, compose, healthcheck, lockfile]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-23T17:39:22Z
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
API keys come from the environment or an uncommitted `.env`; the image contains no data, renders or
DICOM. Dependencies pinned in `requirements.lock` (`uv pip compile pyproject.toml --extra dev --universal`).

`deploy.md` is the runbook the fleet sprint's deployer follows (venv install + Compose + smoke test).
Service details: [API Service](/subsystems/api-service.md).

# Not verified

The Compose stack has not been brought up and smoke-tested in this review. Tracked in
[Delivery Plan & Status](/playbooks/delivery-plan-and-status.md).
