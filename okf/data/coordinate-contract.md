---
type: Data Contract
title: Coordinate Contract
description: BBox, Transform and Page types; every box in any output is [x0, y0, x1, y1] in the original page's pixel grid, and every geometric operation is a tracked 3x3 homography.
resource: src/radintake/core/types.py
tags: [coordinates, bbox, transform, homography, provenance]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-24T11:48:24.228Z
sources:
  - src/radintake/core/types.py
  - src/radintake/tools/loader.py
  - src/radintake/tools/image_ops.py
---

# Rules

- `BBox` = `[x0, y0, x1, y1]` pixels, `x0 < x1`, `y0 < y1`.
- Output boxes are always in ORIGINAL page coordinates: the pixel grid as loaded from the file.
  PNG/JPEG: pixels as stored (EXIF orientation is recorded in `page.meta`, not applied). TIFF: each
  frame as stored. PDF: the page rasterised at `loader.pdf_dpi` (200) — PDF points x dpi/72.
- Every geometry change (rotate, deskew, dewarp, crop, resize) is a `Transform` (3x3 homography)
  from its input image to its output image. `Page.transform` composes ORIGINAL -> CURRENT;
  `page.to_original(box)` maps back.
- The same homography idea is used by the degradation library to move ground-truth boxes through
  synthetic damage: [Data Preparation](/data/data-preparation.md).

Used by: [Preprocessing Agent](/subsystems/preprocessing-agent.md), [Extraction Agent](/subsystems/extraction-agent.md),
[Document PHI Agent](/subsystems/phi-agent.md), [Verifier](/subsystems/verifier-agent.md).

# Guardrails & Anti-Regression Checklist

- Never report a box in preprocessed-image coordinates.
- A non-invertible transform (degenerate dewarp) must fail the stage, not produce garbage boxes.
