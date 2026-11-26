---
type: Subsystem
title: Preprocessing Agent
description: Measures page quality and applies only the operations its metrics demand (orientation, dewarp, deskew, denoise, CLAHE), tracking every geometric change so boxes map back to the original page.
resource: src/radintake/agents/preprocess.py
tags: [preprocessing, image-quality, orientation, deskew, dewarp, agent]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-26T06:02:15.459Z
sources:
  - src/radintake/agents/preprocess.py
  - src/radintake/tools/image_ops.py
  - src/radintake/tools/ocr.py
  - tests/unit/test_preprocess.py
  - tests/unit/test_page_detect.py
  - configs/default.yaml
---

# Policy

Stage `preprocess` (order 20, `failure_flag: preprocess_failed`). A rule policy, not an LLM: the
decisions are small, numeric and must be reproducible per page (`decision <- metric > threshold`).

```
page quad detected (photos)          -> 4-point perspective dewarp (contour detector, else line fallback)
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

# Page detection (contour first, line fallback)

`image_ops.detect_page_quad` returns `(quad, detector, score, contour_rejected)`:

1. **Contour** (`find_page_quad`): Otsu threshold of the whole photo, largest bright region, 4-point
   approximation, global `background / page < quad_min_bg_contrast (0.6)`.
2. **Edge check**: the contour quad is kept only if every side lies on a paper edge
   (`quad_side_support >= quad_contour_min_side_support`, 0.5). A lighting gradient makes one side of
   the page darker than the desk on the other, so the global threshold either finds nothing or merges
   a bright desk corner into the page (a WRONG quad, worse than none: dev 64 of 87 photo pages).
3. **Line fallback** (`find_page_quad_lines`, `quad_fallback: true`): close away dark strokes < 17 px
   (text, rules, handwriting) at 800 px, Canny, Hough lines; every quad of two near-horizontal and
   two near-vertical lines is scored by the fraction of its perimeter on an edge (`score`). Accepted
   only if convex, corners 60-120 degrees, area 0.25-0.97 of the frame, mean support >=
   `quad_fallback_min_support` (0.5), and EVERY side is a local paper step (grey just outside / just
   inside <= `quad_fallback_max_side_ratio`, 0.88). The best quad's sides are refit to the edge pixels.
   A scan without a visible paper edge has no such quad, so it keeps the identity transform.

Trace: the `dewarp` decision carries reason `page_quad_found` (contour) or `page_quad_fallback` (lines)
and numbers only: `fallback` 0/1, `contour_rejected` 0/1, `score`, `out_w`, `out_h`.
Do not retry loosening the contour contrast threshold (0.3, 0.15): it found fewer quads (attempts.md).

Measured 2026-11-26 on the full dev set (218 docs, same ids, `eval/results/phone_photo_v2.json`):
phone-photo EM 0.500 -> 0.614, omission 16.0% -> 6.6%, PHI recall 0.763 -> 0.881; clean and poor
scans unchanged (no scan is dewarped). Only 6 of 87 photo pages had NO contour quad; 64 had a wrong
one, which is why the edge check on contour quads matters more than the fallback itself.

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
