---
type: Subsystem
title: Document PHI Agent
description: Detects synthetic identifiers on OCR words with layout anchors, regex and an NER fallback, masks page images and text, keeps a local-only vault, and is the only producer of phi_masked payloads.
resource: src/radintake/agents/phi.py
tags: [phi, privacy, masking, de-identification, agent, trust-boundary]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-27T10:39:55.720Z
sources:
  - src/radintake/agents/phi.py
  - src/radintake/agents/phi_vault.py
  - configs/phi.yaml
  - tests/unit/test_phi_docs.py
  - src/radintake/tools/handwriting.py
  - configs/models.yaml
---

# Role

Stage `phi` (order 30, `failure_flag: phi_uncertain`) runs after preprocessing and BEFORE extraction
or any model call. It is the only code that creates a `MaskedPayload` stamped `phi_masked=True`;
the [Guarded Model Client](/subsystems/model-client-guard.md) refuses every other payload.

# Layered detector (`configs/phi.yaml`)

1. **Layout anchors** — a label on an OCR line (`Patient Name:`, bare `Patient:` (colon required,
   added 2026-11-25), `Patient ID`, `MRN`, `DOB` /
   `Date of Birth`, `Accession`); the value is the tokens to its right up to the next label. Gives
   typed header fields. The same anchors drive header extraction in the [Extraction Agent](/subsystems/extraction-agent.md).
2. **Regex** — MRN / accession / date formats anywhere on the page.
3. **Names** — honorific ("Dr. X") and credential sign-off rules ("First Last, MD", "G. Rodriguez, RT")
   always run; Presidio NER is added when installed. A `REFERRING_PHYSICIAN` anchor covers "Referring:".
   (Technologist names had recall 0 before the credential rule learned `RT` and initials.)
   A credential-anchored rule also takes whatever sits between a dash / "by" and RT/MD/RN as a name,
   because OCR of handwritten sign-offs rarely yields a clean "Initial. Last" ("— L. Broo ks, RT").
   Dev (2026-11-24): document PHI recall 0.83 / precision 0.81; technologist 0.52, radiologist 0.74 -
   the remaining misses are mostly handwritten sign-offs OCR cannot read (see the step inspector).
4. **Handwritten sign-offs** (`HandwritingPass`, `phi.handwriting` in `configs/phi.yaml`, added
   2026-11-27) - runs AFTER layers 1-3, inside `mask_page`, on the processed page with every rule
   detection already filled. `select_handwriting_crops` picks only (a) credential anchors: the
   unmasked words left of an `RT` / `MD` / ... token back to a separator (dash, `=`, `by`) or a
   sentence end, at most `max_name_words`, continuing onto the previous line when the name wrapped;
   plus the unmasked value right of a sign-off label (`Technologist`, `Signed by` ...) when the line
   has no credential; and (b) lines in the sign-off zone (from the first anchor line down, else the
   bottom `zone_frac` of the page) with a word below `low_conf`. Each segment is read by the local
   handwriting model (`models.hw_reader`, TrOCR-small-handwritten) through the guarded reader (see
   the guard decision in [Guarded Model Client](/subsystems/model-client-guard.md)); the text (with
   `— ` prefixed when a dash preceded the name) goes back through `PhiDetector`, and each name span
   is mapped proportionally onto the segment box and snapped to its OCR words, which are masked
   (one `PERSON` detection per crop, source `handwriting`, bbox in original coordinates).
   `enabled: auto`: used only when `transformers` + `torch` are installed (extra `[handwriting]`,
   not in `requirements.lock`) AND the weights are cached; otherwise a `phi_handwriting` trace event
   records `status: skipped` and the reason (`transformers_missing`, `torch_missing`,
   `weights_not_cached`, `model_calls_forbidden`, `disabled`) and the output is unchanged. A reader
   error mid-case also only records a skip. Per-page `phi_handwriting` events carry crop / read /
   detection counts, `read_ms` and the model id - never recognised text.
Uncertain spans raise `phi_uncertain` (case -> review).

Words come from the `tables` stage cache when present (PDF text layer, or OCR with re-read table cells),
so digital PDFs are masked on exact word boxes: [PDF Text Layer & Table Reading](/subsystems/tables-and-text-layer.md).

# Masking and the vault

- Image: detected boxes are filled on a COPY of the page; masked page images are written as
  `phi.masked_artifacts` in the output.
- Text: spans become typed placeholders (`[PATIENT_NAME_1]`).
- `PhiVault` (`agents/phi_vault.py`) maps placeholders back to values LOCALLY so the structured
  output can still carry header fields. It is never serialised to outputs or traces, and every
  value put in it is registered with the log/trace scrubber ([Tracing & PHI-safe Logging](/subsystems/tracing-and-logging.md)).

# Guardrails & Anti-Regression Checklist

- No external call may happen before this stage; keep stage order `phi` < `extract`.
- The handwriting reader must stay `kind: local`, only ever receives `PhiCrop`s cut after the rules
  ran, and must never download weights; it never fails the stage.
- Never construct `MaskedPayload(phi_masked=True)` outside this module.
- Every value entering the vault must be registered with `register_phi`.
- Integration tests assert header regions are near-uniform after masking and that no synthetic PHI string appears in stdout, stderr or any file under `_traces`.
