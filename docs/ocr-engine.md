# OCR engine choice

Tesseract 5 (LSTM, `--oem 1 --psm 6`) via `pytesseract`, wrapped in `radintake.tools.ocr`.

* CPU only: no GPU memory used, so the 4 GB RTX 3050 stays free for the VLM. CI installs `tesseract-ocr` with apt.
* Offline: local binary, no network; deterministic (`OMP_THREAD_LIMIT=1`).
* Measured (Windows laptop CPU, 1700x2200 page): `orientation()` ~0.8 s, `ocr()` ~1-2 s per page.
* `orientation()` returns the clockwise rotation (0/90/180/270) that makes the page upright; OSD first, rotation scoring (summed word confidence of 4 rotations) when OSD is unsure.
* Local setup: Windows `winget install UB-Mannheim.TesseractOCR` (binary auto-detected, or set `TESSERACT_CMD`); Docker image needs `apt-get install tesseract-ocr` (noted for the Dockerfile).
