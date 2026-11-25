---
type: Subsystem
title: Preprocessing Agent
description: Measures page quality and applies only the operations its metrics demand (orientation, dewarp, deskew, denoise, CLAHE), tracking every geometric change so boxes map back to the original page.
resource: src/radintake/agents/preprocess.py
tags: [preprocessing, image-quality, orientation, deskew, dewarp, agent]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-25T08:12:27.009Z
sources:
  - src/radintake/agents/preprocess.py
  - src/radintake/tools/image_ops.py
  - src/radintake/tools/ocr.py
  - tests/unit/test_preprocess.py
---

# Policy

Stage `preprocess` (order 20, `failure_flag: preprocess_failed`). A rule policy, not an LLM: the
decisions are small, numeric and must be reproducible per page (`decision <- metric > threshold`).

```
page quad detected (photos)          -> 4-point perspective dewarp
background unevenness > 20           -> flat-field illumination correction (divide by background)
orientation (tools/ocr.orientation)  -> lossless rotate 90/180/270, judged on the flat page
|skew| > deskew_min_deg (0.3)        -> rotate by measured skew (capped at deskew_max_deg 15)
noise > noise_sigma_max (6.0)        -> non-local-means denoise
contrast < contrast_min (120)        -> CLAHE
```

Metrics and ops live in `tools/image_ops.py` (Laplacian variance, noise sigma, contrast, skew
angle, page-quad detection). Each geometry op returns `(new_image, Transform)` and never modifies
its input. Thresholds: [Configuration](/operations/configuration.md). Decisions and metric values
are written to the trace ([Tracing](/subsystems/tracing-and-logging.md)).

# Coordinates

The source image is never overwritten. The page's composed `Transform` maps ORIGINAL -> CURRENT
pixels; `map_to_original` / `Page.to_original` send any box found later back to the original grid.
See [Coordinate Contract](/data/coordinate-contract.md). Unit tests check a 2 px round trip.

# Why this order (measured 2026-11-24)

The first dev run gave phone photos mean EM 0.07 with 82% of fields omitted although the cleaned
images were readable: a lighting gradient (background p95-p5 of 30-70 grey levels vs ~1 for a scan)
made Tesseract's single global threshold turn half the page into noise, and orientation was judged on
the skewed photo before dewarping (an upright page was turned 90°). Dewarp -> flat-field -> orientation
lifted photos to EM 0.43 / omission 0.12 and photo PHI recall from 0.32 to 0.82 on a 30-doc dev sample.

# Guardrails & Anti-Regression Checklist

- Every new geometric op must return a `Transform`; an untracked resize/crop silently breaks provenance.
- Photo-case provenance is in the photo's own (original) pixel grid, not the dewarped one.
- Orientation relies on Tesseract; without it the stage degrades (see [CI & Test Strategy](/operations/ci-and-test-strategy.md) for the local PATH issue).
