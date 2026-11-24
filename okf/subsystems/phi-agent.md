---
type: Subsystem
title: Document PHI Agent
description: Detects synthetic identifiers on OCR words with layout anchors, regex and an NER fallback, masks page images and text, keeps a local-only vault, and is the only producer of phi_masked payloads.
resource: src/radintake/agents/phi.py
tags: [phi, privacy, masking, de-identification, agent, trust-boundary]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-24T12:24:50.803Z
sources:
  - src/radintake/agents/phi.py
  - src/radintake/agents/phi_vault.py
  - configs/phi.yaml
  - tests/unit/test_phi_docs.py
---

# Role

Stage `phi` (order 30, `failure_flag: phi_uncertain`) runs after preprocessing and BEFORE extraction
or any model call. It is the only code that creates a `MaskedPayload` stamped `phi_masked=True`;
the [Guarded Model Client](/subsystems/model-client-guard.md) refuses every other payload.

# Layered detector (`configs/phi.yaml`)

1. **Layout anchors** — a label on an OCR line (`Patient Name:`, `Patient ID`, `MRN`, `DOB` /
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
Uncertain spans raise `phi_uncertain` (case -> review).

# Masking and the vault

- Image: detected boxes are filled on a COPY of the page; masked page images are written as
  `phi.masked_artifacts` in the output.
- Text: spans become typed placeholders (`[PATIENT_NAME_1]`).
- `PhiVault` (`agents/phi_vault.py`) maps placeholders back to values LOCALLY so the structured
  output can still carry header fields. It is never serialised to outputs or traces, and every
  value put in it is registered with the log/trace scrubber ([Tracing & PHI-safe Logging](/subsystems/tracing-and-logging.md)).

# Guardrails & Anti-Regression Checklist

- No external call may happen before this stage; keep stage order `phi` < `extract`.
- Never construct `MaskedPayload(phi_masked=True)` outside this module.
- Every value entering the vault must be registered with `register_phi`.
- Integration tests assert header regions are near-uniform after masking and that no synthetic PHI string appears in stdout, stderr or any file under `_traces`.
