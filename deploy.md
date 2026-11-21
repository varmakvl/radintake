# deploy.md — deploy runbook (used by the fleet-sprint `deployer`)

Target: the local member machine (Windows, Git Bash / PowerShell). The service is deployed
two ways: an editable install into `.venv` (always) and the Docker Compose stack (once
`docker-compose.yml` exists in the repo root).

## Permissions

- `python`
- `.venv/Scripts/python`
- `docker compose`
- `curl`

## Deploy

1. `python -m venv .venv` (skip if `.venv` exists)
2. `.venv/Scripts/python -m pip install -q -r requirements.lock` if `requirements.lock` exists, else `.venv/Scripts/python -m pip install -q -e ".[dev]"`
3. `.venv/Scripts/python -m pip install -q -e . --no-deps`
4. If `docker-compose.yml` exists: `docker compose up -d --build --wait`

## Smoke test

1. `.venv/Scripts/python -m radintake.cli --help` exits 0.
2. If `docker-compose.yml` exists: `curl -fsS http://localhost:8000/health` returns JSON with `"status":"ok"`.

## CI

GitHub Actions workflow `.github/workflows/ci.yml`: lint + unit tests + integration tests
(`RADINTAKE_OFFLINE=1`) + eval regression gate against `eval/baseline.json`.

## Teardown

If `docker-compose.yml` exists: `docker compose down`.
