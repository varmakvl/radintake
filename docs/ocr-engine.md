# OCR engine choice

Tesseract 5 (LSTM, `--oem 1 --psm 6`) via `pytesseract`, wrapped in `radintake.tools.ocr`.

* CPU only: no GPU memory used, so the 4 GB RTX 3050 stays free for the VLM. CI installs `tesseract-ocr` with apt.
* Offline: local binary, no network; deterministic (`OMP_THREAD_LIMIT=1`).
* Measured (Windows laptop CPU, 1700x2200 page): `orientation()` ~0.8 s, `ocr()` ~1-2 s per page.
* `orientation()` returns the clockwise rotation (0/90/180/270) that makes the page upright; OSD first, rotation scoring (summed word confidence of 4 rotations) when OSD is unsure.
* Local setup: Windows `winget install UB-Mannheim.TesseractOCR` (binary auto-detected, or set `TESSERACT_CMD`); Docker image needs `apt-get install tesseract-ocr` (noted for the Dockerfile).

## Engine switch: Tesseract (default) or PaddleOCR (optional)

`ocr.engine` in `configs/default.yaml` (`tesseract` | `paddle`), overridden by a `--config` YAML and
then by the environment variable `RADINTAKE_OCR_ENGINE` (same precedence as every other key in
`radintake.config`). `radintake.pipeline.process_file` pins the run's engine with
`radintake.tools.ocr.set_engine(cfg.ocr_engine)`; direct callers of `ocr()` resolve
env > config files. `orientation()` is always Tesseract.

* **PaddleOCR backend** (`radintake.tools.ocr_paddle`): PaddleOCR 2.10 `ocr_version="PP-OCRv4"`, English
  (`en_PP-OCRv4_rec` recogniser + `en_PP-OCRv3_det` detector, the pair PaddleOCR ships for English; Apache-2.0),
  CPU, no angle classifier, MKL-DNN off. It returns the same `OcrWord` contract as Tesseract:
  boxes in the pixel grid of the input image (clamped to it), `conf` in 0..1 (the recogniser's line
  score, shared by the line's words), and `block/par/line/word` ids. PaddleOCR detects *lines*, so
  detected boxes whose vertical centres fall in one another's span are merged into one row (one
  `line_id`, like Tesseract `--psm 6`), rows go top to bottom and words left to right, and each
  word gets the slice of its line box proportional to its character offsets (uniform character
  advance). `block = par = 1`.
* **Install** (never part of CI or the default Docker image):
  `pip install -e .[paddle] -r requirements-paddle.lock`. Pins live in `requirements-paddle.lock`
  (paddlepaddle 3.0.0, paddleocr 2.10.0 and their extra dependencies on top of
  `requirements.lock`); `requirements.lock` is untouched.
* **Weights**: PaddleOCR caches PP-OCRv4 det/rec weights in `~/.paddleocr` on first use. Under
  `RADINTAKE_OFFLINE=1` nothing is downloaded: prefetch once with
  `python -m radintake.tools.ocr_paddle --fetch`, or set `RADINTAKE_PADDLE_MODEL_DIR` to a folder
  holding `det/` and `rec/` inference models.
* **Missing extra -> clear error, no fallback.** With `ocr.engine: paddle` and the extra (or the
  cached weights offline) missing, `ocr()` raises `OcrUnavailable` naming the extra. A silent
  Tesseract fallback was rejected: it would let a comparison run labelled PaddleOCR measure
  Tesseract. In the pipeline the error surfaces like a missing Tesseract binary (stage error).
* **Traces**: `configs/models.yaml` registers `ocr_paddle` (`paddleocr-ppocrv4`, kind local). The
  config loader exposes the selected engine's entry as `models.ocr`, so the run manifest's
  `model_ids["ocr"]` is `tesseract-5` or `paddleocr-ppocrv4`, and every `ocr` `tool_call` event
  carries `tool_version` (`tesseract 5.x` or `paddleocr 2.10.0 PP-OCRv4 (paddle 3.0.0)`).

## Trial result: Tesseract stays the default (dev evidence)

Same 60 dev documents (20 per tier, seed 140), OCR-first, `RADINTAKE_OFFLINE=1`, both on CPU, run one
after the other ([`eval/results/ocr_engine_comparison.json`](../eval/results/ocr_engine_comparison.json)):

| Engine | EM clean / poor / photo | CER | PHI recall / precision | p50 / p95 per page | GPU |
|---|---|---|---|---|---|
| Tesseract 5.4 | **0.861 / 0.511 / 0.547** | 0.157 | 0.889 / **0.775** | **3.6 s / 4.8 s** | 0 |
| PaddleOCR 2.10 PP-OCRv4 | 0.792 / 0.402 / 0.413 | **0.107** | **0.937** / 0.714 | 6.5 s / 10.3 s | 0 |

Engine choice can only be keyed on what is known at run time (the router's input class, not the
quality tier). PaddleOCR loses exact match in every input class (scan -3.8, photo -13.5, pdf -13.6,
tiff -7.6 points), so no `engine_by_input_class` key was added and the default stays Tesseract
everywhere; the Docker image gets no paddle build arg. Main cause: PP-OCRv4 often omits the space
after label punctuation (`Accession No.ACC-11144883`), so the OCR-first label parser, tuned on
Tesseract tokens, keeps label text in the value. Its better character accuracy on photos and higher
PHI recall make it a candidate again only if the label parser learns to split glued labels.

