# radintake

PHI-safe, multi-agent intake of scanned radiology reports and DICOM images: structured,
source-cited extraction, a human-review queue, cited retrieval / QA, and an evaluation harness
with a CI regression gate.

> Status: **in development.** See [`PLAN.md`](PLAN.md) for the design and
> [`requirements.md`](requirements.md) for scope. The end-to-end contract is the integration
> suite in [`tests/integration/`](tests/integration/).

## Data licensing

The IU Chest X-ray collection (Open-i) is CC BY-NC-ND 4.0. **No rendered pages or raw data are
committed to this repository**: only the scripts, seeds, IDs and split files needed to regenerate
them. All test fixtures are synthetic and generated at test time.

## Quick start (dev)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Windows; use .venv/bin/python on Linux/macOS
.venv/Scripts/python -m pytest tests/unit -q
RADINTAKE_OFFLINE=1 .venv/Scripts/python -m pytest tests/integration -m integration -q
```

## Run the service (Docker Compose)

```bash
docker compose up -d --build --wait      # CPU default; API on http://localhost:8000
curl -fsS http://localhost:8000/health   # {"status":"ok",...}
docker compose --profile ocr-gpu up -d --build --wait   # optional NVIDIA variant on :8001
docker compose down
```

Data and traces live in the `radintake-data` volume (`/data`). API keys are read from the
environment or an uncommitted `.env` file; the image contains no `data/`, renders or DICOM files.
`requirements.lock` pins the dependencies (`uv pip compile pyproject.toml --extra dev --universal -o requirements.lock`).

## Batch CLI

```bash
radintake batch --in <folder> --out <folder>
```

Writes one `<input-name>.json` per input, following [`schemas/cli_output.v1.json`](schemas/cli_output.v1.json).
Unsupported or corrupt files get a JSON with `status: "error"` and never stop the batch.

## Repository layout

See `PLAN.md` § 0. Sections for data download, service, CI gate and reproducing the report's
numbers will be added as they land.
