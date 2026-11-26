---
type: Subsystem
title: Verifier, Injection Detector & Review Queue
description: Checks every extracted value against the source, validates the schema, flags omissions and prompt-injection text, and routes doubtful cases to an append-only human review queue without ever changing a value.
resource: src/radintake/agents/verify.py
tags: [verifier, grounding, hallucination, omission, prompt-injection, review-queue]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-26T12:43:30.238Z
sources:
  - src/radintake/agents/verify.py
  - src/radintake/agents/injection.py
  - src/radintake/store/review_queue.py
  - tests/unit/test_verify.py
  - tests/unit/test_injection_review.py
---

# Verifier (stage `verify`, order 50)

Fixed checks plus routing; it NEVER changes, replaces or invents a value. It only sets
`grounded` / `needs_review` / `low_confidence` on fields and raises case flags.

- **Schema**: report.v1 validation -> `schema_error`.
- **Grounding**: each non-empty value is `rapidfuzz.partial_ratio`-matched (>= `ground_min_ratio` 90)
  against the OCR text of its cited box plus `box_margin_px` (40), then against the whole document
  -> `grounded`; an ungrounded value is kept but flagged `ungrounded_value` + `needs_review`.
  This is the hallucination guard.
- **Omission**: a section heading seen on a page while the field is empty -> `omission`.
- **Cross-field sanity**: ID patterns, dates.
- Low confidence (< 0.5) -> `low_confidence` + `needs_review`.

# Injection detector (stage `injection`, order 55)

Local patterns plus a word-count heuristic, no model call. Instruction-like text in extracted fields
or page OCR raises `prompt_injection_suspected` (case -> `needs_review`) and marks fields
`needs_review`. The text stays exactly as extracted: it is data, never obeyed or removed.

The OCR-tolerant fuzzy pass squashes spacing and punctuation, but never across a clause boundary: text is split on `.` `;` `:` first and each clause is matched alone, so "Pacemaker system; prompt evaluation" is clean. The bare phrase "new instructions" only counts at the start of a clause (regex still catches "new instructions:"). `RetrievalIndex(config=...).add_case` applies `injection.fuzzy_threshold` / `extra_patterns` from config, so `fuzzy_threshold: null` also stops the fuzzy pass dropping sentences from the QA index.

# Review queue

`store/review_queue.py`: append-only SQLite table under `RADINTAKE_DATA_DIR` holding ids, flags,
reason codes and field NAMES only — never PHI or values. The API exposes it at `GET /review-queue`
([API Service](/subsystems/api-service.md)); batch runs also append `<out>/review_queue.jsonl`.

# Guardrails & Anti-Regression Checklist

- "Fix by guessing" is forbidden: a doubtful value goes to review, not to a model.
- Injected pages must keep their true impression and be flagged (integration test).
- Review-queue rows must never carry values or PHI.
