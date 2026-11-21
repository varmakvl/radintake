# requirements.md — sprint requirements (planner input)

Product: **radintake** — a small multi-agent system that turns messy radiology inputs (scanned /
photographed report pages, PDF/TIFF, DICOM) into PHI-safe, structured, source-cited data that a
clinician can query, packaged as a service with an evaluation harness and CI regression gate.

The detailed design and day plan is in `PLAN.md` (binding unless a task documents why it deviates).
The end-to-end contract is `tests/integration/` (see `integ-test-playbook.md`) — the sprint is done
only when that suite passes with `RADINTAKE_OFFLINE=1` and the deliverables below exist.

## Constraints (read first — riskiest assumptions)

1. **Hardware:** member machine has an RTX 3050 Laptop GPU with **4 GB VRAM**, Windows 11, Python 3.11.
   Local models must fit (PaddleOCR / docTR / Tesseract / TrOCR-small/base / Florence-2-base / small embedders).
   Any large VLM is reached via API, **only after the PHI agent has masked the payload**.
2. **Offline mode:** `RADINTAKE_OFFLINE=1` must run the whole pipeline with local models only
   (used by CI and the integration tests). API VLM is an optional enhancement behind a flag.
3. **Licences:** IU Open-i is CC BY-NC-ND — rendered pages and raw data are NEVER committed
   (the repo is public). Commit scripts, IDs, seeds, split files, metrics and traces only
   (traces must contain no PHI and no full report text beyond what tests need — prefer hashes).
4. **Data access:** IU reports+images from Open-i (public download). Pseudo-PHI-DICOM-Data from TCIA
   (DOI 10.7937/s17z-r072) via NBIA Data Retriever or the public TCIA/NBIA REST API.
   The download step is the riskiest external dependency: Task 1 must prove both downloads work
   (script + manifest + checksums) or document the fallback.
5. **Seeds:** synthetic identities use Faker with **seed 140**; all randomness is seeded and recorded.

## Priority order (map to bead priorities)

P1 — Data & answer key:
- Render >= 300 IU reports as realistic report pages (hospital header with synthetic name, patient ID,
  DOB, accession no.; exam-details table; Indication/Comparison/Findings/Impression); include
  multi-page documents.
- >= 30% of pages have 1–2 handwritten-style fields (handwriting fonts or synthesis).
- Three quality tiers: clean scan, poor scan, phone photo — rotation incl. 90/180, skew, perspective,
  blur, noise, low DPI, JPEG, uneven lighting/shadows, stamps/lines. Outputs PNG/JPEG, multi-page PDF,
  multi-page TIFF. Ground truth for every field incl. bbox (kept consistent through degradations).
- >= 10 adversarial pages with injected instructions in report text.
- Near-duplicate detection; split by report into dev and frozen golden test set (>= 100 pages),
  stratified by tier, dup groups never cross; save IDs, seeds, dedup method.
- >= 40 DICOM instances across modalities incl. burned-in text; GT derived from TCIA curated
  counterpart + mapping files; document derivation and incompleteness.

P1 — Agents (extraction, verification, PHI):
- Orchestrator: classify input (DICOM / scan / photo / PDF / TIFF), plan steps, step limits,
  timeouts, retries. Document which steps are fixed pipeline vs agent and why.
- Preprocessing agent: quality metrics -> chooses orientation / deskew / dewarp / denoise / contrast;
  never overwrites source; coordinates map back to original page.
- PHI agent: runs before any external call. Documents: detect + mask synthetic identifiers.
  DICOM: de-identify header; detect + redact burned-in pixel text without altering the rest;
  output stays valid DICOM.
- Extraction agent: schema-valid JSON (header fields, Indication, Comparison, Findings, Impression)
  with per-field confidence + provenance (page, bbox). Compare OCR-first, VLM-first, hybrid on dev set;
  choose on evidence.
- Verifier agent: value-in-source check, schema validation, omission flags, low-confidence /
  failed items -> human-review queue (never guess).

P2 — Harness & evaluation:
- Golden QA set >= 25 questions (MeSH as relevance labels where possible), >= 5 unanswerable.
- Replayable traces per run (input hash, model, prompt + tool versions, agent steps, tool calls,
  outputs, timings); re-score a past run from traces without calling any model.
- Prompts, tool schemas, model ids versioned in repo and recorded in every trace.
- CI regression gate (GitHub Actions): tests + eval subset vs stored baseline thresholds;
  include one commit where the gate fails on purpose (and the fix after it).
- Metrics on golden set: per-field EM, CER, hallucination %, omission % (overall + per tier);
  PHI recall/precision + every residual case; DICOM validity after redaction; retrieval recall@5,
  citation accuracy, answer correctness, correct abstention; injected-page behaviour; % to human review;
  latency p50/p95 per page; GPU memory; cost per page.
- Error analysis of the 10 worst golden pages with failure categories.

P2 — Retrieval / QA agent: hybrid retrieval over verified extractions; answers cite document,
page, bbox; abstains when evidence is insufficient; document text is data, never instructions.

P2 — Deployment: FastAPI in Docker Compose, one command: `POST /cases`, `GET /cases/{id}`,
`GET /review-queue`, `POST /query`, `GET /health`. Batch CLI `radintake batch --in --out` (one JSON per
input, schema `schemas/cli_output.v1.json`, documented in README) — must work on an unseen folder.
Structured logging with no PHI, pinned deps (`requirements.lock`), health checks, clear errors.

P3 — Bonus: LoRA/QLoRA fine-tune of a small OCR/VLM (fits 4 GB, e.g. Florence-2-base or TrOCR) on dev
data; compare with base and best pipeline on quality, latency, cost.

P3 — Report & README: report (DOCX/PDF) with architecture diagram, results tables (overall + per tier),
pipeline comparison, chronological steps incl. failed attempts, PHI approach + residual cases,
injection results, error analysis, best/median/worst overlays, DICOM before/after, tech details
(models, prompt versions, hardware, latency, GPU mem, cost/page, seeds, lib versions), trace
screenshots. README: setup, data download + generation, service, CLI, local CI gate, reproduce
every number, repo layout.

## Definition of done

- `pytest tests -m integration` passes with `RADINTAKE_OFFLINE=1`.
- CI green on the PR (with the intentional red commit visible in history).
- Golden-set numbers produced once, traces committed under `traces/final/`, re-scorable offline.
- No licensed data or PHI in the repo.
