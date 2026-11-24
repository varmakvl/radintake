# radintake

PHI-safe, multi-agent intake of scanned radiology reports (clean scans, poor scans, phone photos;
PNG/JPEG/PDF/TIFF) and DICOM images. It produces structured, source-cited extractions, a human-review
queue, cited retrieval / QA with abstention, and an evaluation harness with a CI regression gate.
**Every model is open source and runs locally** (Tesseract 5, Qwen2.5-VL-3B and Qwen2.5-3B via Ollama,
Florence-2, all-MiniLM-L6-v2), so the cost per page is $0 and no page leaves the machine.

- Design and decisions: [`PLAN.md`](PLAN.md), [`docs/agents_vs_fixed.md`](docs/agents_vs_fixed.md),
  [`docs/ocr-engine.md`](docs/ocr-engine.md). The living architecture bundle is [`okf/index.md`](okf/index.md).
- Scope: [`requirements.md`](requirements.md). The end-to-end contract is [`tests/integration/`](tests/integration/).
- Results: [Results](#results) below; the full report is built by `docs/report/build_report.py` (see [Report](#report)).

## Architecture in one paragraph

A fixed pipeline with agentic branches. The **router** types the input; the **planner** runs stages
under step / time / retry budgets and records every decision in a PHI-free trace. Pages are
**preprocessed** (dewarp, illumination flattening, orientation, deskew, denoise/CLAHE) into derived
images while every box stays in original-page pixels. The **PHI agent** (layout anchors + regex +
NER) masks the page **before any model sees it**; `ModelClient` refuses an unmasked page. DICOM goes
through **dicom_deid** (profile-driven header actions, UID remapping, burned-in text redaction).
**Extraction** is OCR-first by default (hybrid and VLM-first are switchable), then **verify** grounds
each value in the OCR text and **injection** checks for instructions hidden in report text. Low
confidence goes to the **review queue**; verified fields go to the **retrieval index** (SQLite FTS5
BM25 + embeddings, RRF) behind a **QA agent** that answers only with verified citations (document,
page, bbox) and abstains otherwise. Storage is one SQLite file holding JSON documents keyed by a
pseudonymous `subject_key`; there is no vector database (the corpus is small and FTS5 + in-row
embeddings are enough).

## Data licensing

The IU Chest X-ray collection (Open-i) is **CC BY-NC-ND 4.0**. No rendered pages or raw data are
committed: only scripts, seeds, ids and split files needed to regenerate them. Score records and
committed traces hold ids, numbers and sha256 hashes, never values. Test fixtures are synthetic and
generated at test time. TCIA Pseudo-PHI-DICOM-Data is used locally and never committed.

## Setup

Python 3.11, Tesseract 5 on `PATH`, and [Ollama](https://ollama.com) for the VLM / answer model.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock          # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pip install -e ".[dev]" --no-deps
ollama pull qwen2.5vl:3b && ollama pull qwen2.5:3b                # only for hybrid / VLM-first / LLM answers
```

`RADINTAKE_OFFLINE=1` uses local models only (the default in Docker and CI). Hardware used for every
number below: RTX 3050 Laptop 4 GB, Windows 11; the default OCR-first pipeline needs no GPU.

## Data download and generation

```bash
pip install -e ".[data]" && playwright install chromium
python -m data_prep.download iu --no-images   # 3,955 IU XML reports -> data/iu (sha256 manifest)
python -m data_prep.iu_parse                  # -> data/iu/reports.jsonl
python -m data_prep.audit                     # -> data_prep/audit.md (aggregates only)
python -m data_prep.fonts                     # handwriting fonts (OFL/Apache, sha256-pinned)
python -m data_prep.render                    # 330 docs, synthetic identities (Faker seed 140)
python -m data_prep.generate                  # 3 degradation tiers, PNG/JPEG/PDF/TIFF + box-exact GT
python -m data_prep.split                     # near-dup groups + dev/golden split -> splits/ (committed)
python -m data_prep.dicom_gt                  # TCIA evaluation/curated pairs -> DICOM ground truth
```

TCIA Pseudo-PHI-DICOM-Data is downloaded by hand with the NBIA Data Retriever into `data/tcia/`
(both the evaluation and the curated halves). `splits/golden.lock` pins the golden list by hash.

## Service (Docker Compose)

```bash
docker compose up -d --build --wait      # API on http://localhost:8000 (CPU)
curl -fsS http://localhost:8000/health
docker compose --profile ocr-gpu up -d --build --wait   # optional NVIDIA variant on :8001
docker compose down
```

| Endpoint | Purpose |
|---|---|
| `POST /cases` | upload a file (multipart); returns the case id |
| `GET /cases/{id}` | status, extraction with confidence and provenance, review flags |
| `GET /review-queue` | cases and fields waiting for a human |
| `POST /query` | `{"question": ...}` → cited answer or abstention |
| `GET /health` | liveness, model availability |

The VLM runs in Ollama on the host (`OLLAMA_HOST`, default `http://host.docker.internal:11434`).
Data and traces live in the `radintake-data` volume. Logs are structured JSON with no PHI.

## Batch CLI

```bash
radintake batch --in <folder> --out <folder> [--config my.yaml]
```

Writes one `<input-name>.json` per input following [`schemas/cli_output.v1.json`](schemas/cli_output.v1.json):
`status`, `input_type`, `report` fields (value, confidence, provenance `{page, bbox, source}`,
`needs_review`), PHI detections (types and boxes, never values), `injection` flags and `trace_id`.
Unsupported or corrupt files get `status: "error"` and never stop the batch.

## Seeing what each step does

```bash
python -m radintake.inspect <file-or-folder> --out runs/inspect [--pipeline hybrid]
```

Writes `runs/inspect/index.html`: per input, the original, preprocessed and PHI-masked page, the OCR
words, the VLM input and reply (if any), the extraction with provenance boxes and the trace.
DICOM inputs show the pixel data before and after redaction. Output is local only (it shows PHI).

## Evaluation and the CI gate

```bash
python -m radintake.eval.run --split dev [--pipeline ocr_first|vlm_first|hybrid] [--sample-per-tier K] [--dicom]   # QA set scored by default
python -m radintake.eval.run --split golden --final --dicom        # once only; writes eval/GOLDEN_USED.json
python -m radintake.eval.error_analysis --run runs/<run>            # 10 worst pages, categories, overlays
python -m radintake.eval.rescore --traces traces/final --check      # re-score with every model call forbidden
python -m radintake.eval.gate --subset ci                           # the CI regression gate, locally
python -m radintake.eval.versions                                   # check the prompt/schema/config lock
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs lint, unit tests, the offline
integration suite and the gate against [`eval/baseline.json`](eval/baseline.json) on a licence-free
synthetic subset. Commit `f5d4033` breaks the patient-name PHI anchor on purpose (CI red) and
`19c9d6e` reverts it (CI green).

## Results

<!-- RESULTS:START -->
Single golden run `20261124T170304Z-a5a6c448` (OCR-first, 112 documents / 145 pages,
scored once; record in [`traces/final/`](traces/final/), re-scorable offline).

| Tier | Field EM | CER | Hallucination | Omission | PHI recall |
|---|---|---|---|---|---|
| **Overall** | 0.672 | 0.169 | 8.8% | 7.6% | 83.1% |
| Clean scan | 0.884 | 0.016 | 0.9% | 0.0% | 91.0% |
| Poor scan | 0.579 | 0.218 | 8.7% | 9.7% | 80.8% |
| Phone photo | 0.549 | 0.279 | 18.4% | 13.3% | 77.7% |

- **PHI (documents):** recall 83.1%, precision 80.7%; 149 of 883 instances missed,
  mostly technologist / radiologist names and patient names on photos (all listed by id in `metrics.json`).
- **DICOM (48 TCIA files):** 100.0% valid after redaction, header PHI recall 100.0%, UIDs remapped
  100.0%, burned-in text-pixel recall 61.5% (precision 54.4%).
- **Retrieval / QA (32 questions):** recall@5 0.549, citation accuracy 100.0%,
  answer correctness 71.9%, correct abstention 96.9%.
- **Injection:** 4 injected documents, 1 flagged, 0 changed behaviour, 0 false flags.
- **Operations:** 59.8% of documents to human review; latency p50 2.3 s / p95 3.5 s
  per page on CPU; GPU peak 0 MiB; cost $0.00 per page.
- **10 worst pages:** [`eval/results/error_analysis_golden.md`](eval/results/error_analysis_golden.md): 6 of 10 are phone photos and
  OCR reads almost nothing on most of them (on dev this was traced to the page-quad detector missing the paper edge).
<!-- RESULTS:END -->

### Pipeline comparison (dev, 18 documents, 6 per tier)

| Pipeline | Exact match | CER | Hallucination | Omission | p50 / p95 per doc | GPU peak |
|---|---|---|---|---|---|---|
| **OCR-first (default)** | 0.642 | 0.146 | 10.5% | 5.4% | 2.3 s / 3.1 s | 0 |
| Hybrid (VLM re-reads low-confidence crops) | 0.651 | 0.147 | 7.7% | 5.4% | 11.9 s / 35.5 s | 2.9 GB |
| VLM-first (Qwen2.5-VL-3B) | 0.421 | 0.485 | 10.6% | 33.9% | 16.8 s / 35.9 s | 2.9 GB |

OCR-first stays the default: hybrid gains 0.9 points of exact match (all on phone photos) for 5× the
latency. Raw numbers: [`eval/results/pipeline_comparison.json`](eval/results/pipeline_comparison.json);
failed attempts: [`eval/results/attempts.md`](eval/results/attempts.md).

## Reproducing every number

1. Generate the data (above), then `python -m radintake.eval.versions` must pass.
2. Dev: `python -m radintake.eval.run --split dev --dicom` and, for the comparison,
   `--sample-per-tier 6 --pipeline {ocr_first,hybrid,vlm_first}`.
3. Golden (already used once; `eval/GOLDEN_USED.json` records it): re-score the committed traces with
   `python -m radintake.eval.rescore --traces traces/final --check --verify-gt`.

Seeds: data 140 (numpy + Faker), split 140. Model ids, prompt hashes and library versions are written
into every `run.json` and pinned in `eval/VERSIONS.lock` and `requirements.lock`.

## Report

`python docs/report/build_report.py --run runs/golden_final` builds `docs/report/out/report.{html,pdf}` (9 pages:
architecture, data, golden tables per field and tier, pipeline comparison, PHI and residual cases, DICOM before/after,
injection, error analysis with best/median/worst overlays, chronology with failed attempts, technical details, traces).
The output is not committed because it shows rendered IU pages (CC BY-NC-ND); it is shared with the assessor directly.

## Repository layout

```
src/radintake/      agents/ (preprocess, phi, dicom_deid, extract_*, verify, injection, qa),
                    orchestrator/ (router, planner), tools/ (OCR, VLM client, image ops, loader),
                    api/, store/, tracing/, eval/, retrieval.py, inspect.py, cli.py
data_prep/          download, parse, audit, render, degrade, generate, split, DICOM ground truth
configs/            default, models, PHI, DICOM profile, degradation tiers
prompts/, schemas/  versioned and immutable (a change is a new _vN file)
eval/               baseline, version lock, golden and dev QA sets, committed results
splits/             dev / golden ids, golden lock, near-duplicate groups
tests/              unit/ and integration/ (synthetic fixtures only)
okf/                architecture knowledge bundle
```
