# Fixed pipeline steps vs agents

radintake is a fixed pipeline with agentic branches. A step is an **agent** only when it has
to choose between actions based on what it observes, and that choice changes the result. If
a rule is deterministic and testable, it stays a **fixed step**: that is cheaper, faster and
reproducible, and an LLM would only add cost, latency and variance.

| Step | Kind | Where | Why |
|---|---|---|---|
| Input routing (DICOM / PDF / TIFF / scan / photo / unknown) | Fixed | `orchestrator/router.py` | Magic bytes, then extension, then a cheap image heuristic (background unevenness, dark border, camera EXIF). Deterministic, fully unit-tested (a fuzz test checks it never raises). A model adds nothing here. |
| Page loading / rasterisation | Fixed | `tools/loader.py` (stage `load`) | Pure decoding: PDF rasterised at `loader.pdf_dpi` (200), every TIFF frame becomes a page. Page coordinates for a PDF are the raster pixel coordinates. |
| Planning, budgets, retries, timeouts | Fixed state machine | `orchestrator/planner.py` | The plan for each input type comes from the stage registry. Step limits, per-stage timeouts, retries with backoff and the case deadline must hold every time, so this is deliberately not left to a model. |
| Preprocessing (orientation, deskew, dewarp, denoise, contrast) | Agent | `agents/preprocess` (later lane) | Image-quality metrics decide which operations to apply and in what order. Every geometric operation is recorded as a `Transform`, so boxes map back to the original page. |
| PHI detection and masking (documents) | Agent + fixed tools | `agents/phi` (later lane) | Detection mixes patterns, layout and model scores and has to settle uncertain spans (it raises `phi_uncertain`). Masking itself is a fixed tool. It always runs before any external model call; the model client refuses unmasked payloads. |
| DICOM de-identification (header + burned-in pixels) | Fixed header profile + agent for burned-in text | stage `dicom_deid` (later lane) | Header de-identification is a fixed profile (deterministic, auditable). Finding burned-in text needs OCR plus a decision about each region. |
| Extraction (header fields and sections, with provenance) | Agent | `agents/extract` (later lane) | Chooses OCR-first, VLM-first or hybrid per page and field, with confidence and provenance. |
| Verification (grounding, schema, omission, injection) | Fixed checks + routing | `agents/verify` (later lane) | Value-in-source fuzzy match, schema validation and omission checks are deterministic. Cases that fail go to human review, never to a "fix by guessing" model. |
| Output JSON writing and schema validation | Fixed | `pipeline.py` | Every input produces exactly one `cli_output.v1` document. If an output fails schema validation, it is replaced by an `error` document flagged `schema_error`. |
| Tracing | Fixed | `tracing/writer.py` | Every case trace and run manifest is written PHI-free (strings are hashed) so runs can be re-scored offline. |

## How lanes plug in

Stages register themselves with `radintake.orchestrator.stages.register_stage(...)` inside their
own module, under `radintake.agents.*` (or `radintake.stages_ext.*`, or a module listed in the
config key `stages.plugins`). `stages.discover()` imports these modules, so parallel lanes never
edit one shared list. Ordering uses `order` (load = 10; suggested: preprocess 20, phi 30,
dicom_deid 30, extract 40, verify 50).

## Failure semantics (planner)

* An exception in a critical stage (for example `load`) gives status `error` with the message.
* An exception in a non-critical stage gives `needs_review` plus the stage's `failure_flag`
  (for example `preprocess_failed`); the plan continues.
* A timeout, an exhausted step budget or a passed case deadline sets the `timeout` flag. The
  case is `needs_review` if an earlier stage produced partial results, otherwise `error`.
* No failure crashes the batch. The CLI exits 0 whenever the batch ran, and non-zero only for
  invalid arguments.
