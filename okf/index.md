---
okf_version: 0.2
---

# radintake Knowledge Bundle

Architecture knowledge for **radintake**: PHI-safe, multi-agent intake of scanned radiology
reports and DICOM. Start with the [System Overview](/overview.md); the delivery state lives in
[Delivery Plan & Status](/playbooks/delivery-plan-and-status.md).

## Overview
- [System Overview](/overview.md) — End-to-end architecture: fixed pipeline with agentic branches, stage order, trust boundaries.

## Subsystems
- [Orchestrator (router, planner, stage registry, pipeline)](/subsystems/orchestrator.md) — Fixed routing, budgeted state machine, per-input stage plans, one output JSON per input.
- [Preprocessing Agent](/subsystems/preprocessing-agent.md) — Metric-driven orientation, dewarp, deskew, denoise, contrast with tracked transforms.
- [Document PHI Agent](/subsystems/phi-agent.md) — Layered PHI detection, image/text masking, local vault, sole producer of masked payloads.
- [DICOM De-identification Agent](/subsystems/dicom-deid-agent.md) — Basic Profile header actions plus burned-in text redaction; output stays valid DICOM.
- [PDF Text Layer & Table Reading](/subsystems/tables-and-text-layer.md) — Exact words from digital PDFs (hidden-text guard), ruled / rules-only / borderless table detection, per-cell re-read, tables.v1.
- [Extraction Agent](/subsystems/extraction-agent.md) — OCR-first, VLM-first and hybrid pipelines emitting report.v1 with confidence and provenance.
- [Verifier, Injection Detector & Review Queue](/subsystems/verifier-agent.md) — Grounding, schema, omission and injection checks; routes doubtful cases to humans.
- [Retrieval / QA Agent](/subsystems/retrieval-qa-agent.md) — SQLite BM25 + MiniLM hybrid index, extractive cited answers, negation, abstention.
- [Guarded Model Client & VLM Client](/subsystems/model-client-guard.md) — Every model call is offline-checked, PHI-mask-checked, cached and traced.
- [Tracing & PHI-safe Logging](/subsystems/tracing-and-logging.md) — Hashed, replayable JSONL traces and a shared PHI scrubber for logs.
- [API Service](/subsystems/api-service.md) — FastAPI routes, SQLite case store, background processing, problem+json errors.
- [Batch CLI](/subsystems/batch-cli.md) — `radintake batch --in --out`: one schema-valid JSON per input, never crashes on bad files.

## Data
- [Coordinate Contract](/data/coordinate-contract.md) — BBox/Transform/Page types; every output box is in original page pixels.
- [Output & Trace Schemas](/data/output-schemas.md) — cli_output.v1, report.v1, trace.v1 and the Pydantic mirror.
- [Data Preparation](/data/data-preparation.md) — IU/TCIA acquisition, audit, 330 rendered docs with handwriting and injections, 3 tiers, 4 formats, exact GT boxes.
- [Dataset Split & Golden Set](/data/dataset-split-and-golden-set.md) — Near-duplicate groups, hash-locked dev/golden split, DICOM ground truth from TCIA pairs.

## Operations
- [Configuration](/operations/configuration.md) — YAML + env precedence, budgets, thresholds, model registry, offline mode.
- [Deployment](/operations/deployment.md) — Dockerfile, Compose stack, deploy runbook, pinned lockfile.
- [CI & Test Strategy](/operations/ci-and-test-strategy.md) — Unit tests, end-to-end integration contract, GitHub Actions, sprint playbooks.
- [Step Inspector](/operations/step-inspector.md) — Dump every step's input/output (OCR, masking, VLM, extraction, DICOM) for a file; index.html viewer.
- [Evaluation Harness](/operations/evaluation-harness.md) — PHI-free score records, re-scorable runs, version lock, synthetic CI gate, MeSH golden QA.

## Playbooks
- [Delivery Plan & Status](/playbooks/delivery-plan-and-status.md) — Priority order, bead map, what is done, what remains, open risks.

## Concepts
- [SDK Public API](/subsystems/sdk-public-api.md)

## Not yet specified
- `.beads/.local_version` — Undocumented codebase file.
- `.beads/backup/49fcgkkdagptrq9klehrihv9ke6qbllo.darc` — Undocumented codebase file.
- `.beads/backup/49rrd1s9jbc585ta2vqs0j5a2cdhajfa.darc` — Undocumented codebase file.
- `.beads/backup/4v4vt80dhuehmlfn1gc410r5tjq4hthg.darc` — Undocumented codebase file.
- `.beads/backup/52idiuvgqt6diot0tc77trn6n3qsi18p.darc` — Undocumented codebase file.
- `.beads/backup/62jbn35cnibbb69tr324a6vb44j1ugti.darc` — Undocumented codebase file.
- `.beads/backup/6u7vsrkc705tvs1lmuuhaa0rbussjrdq.darc` — Undocumented codebase file.
- `.beads/backup/6vd7cdl0qjjgbvpg45m7dasirqcd1hgk.darc` — Undocumented codebase file.
- `.beads/backup/8fcti2g99n67msacga5bi7s33mpnetc7.darc` — Undocumented codebase file.
- `.beads/backup/backup_state.json` — Undocumented codebase file.
- `.beads/backup/d0m7k32sdq3983k26rhq62j9pnb6vkap.darc` — Undocumented codebase file.
- `.beads/backup/l2v862otk0ik6nu00a7726tmssqmu2ea.darc` — Undocumented codebase file.
- `.beads/backup/LOCK` — Undocumented codebase file.
- `.beads/backup/manifest` — Undocumented codebase file.
- `.beads/backup/od24iq3pg7nrcng5u0mg7pre9so3knoc.darc` — Undocumented codebase file.
- `.beads/backup/tiejp78tte8cv4lnvconthkco9tfrcik.darc` — Undocumented codebase file.
- `.beads/backup/tjdt4hjsbkccp6prgfsbdqtdbhfakrrq.darc` — Undocumented codebase file.
- `.beads/config.yaml` — Undocumented codebase file.
- `.beads/embeddeddolt/.lock` — Undocumented codebase file.
- `.beads/export-state.json` — Undocumented codebase file.
- `.beads/hooks/post-checkout` — Undocumented codebase file.
- `.beads/hooks/post-merge` — Undocumented codebase file.
- `.beads/hooks/pre-commit` — Undocumented codebase file.
- `.beads/hooks/pre-push` — Undocumented codebase file.
- `.beads/hooks/prepare-commit-msg` — Undocumented codebase file.
- `.beads/interactions.jsonl` — Undocumented codebase file.
- `.beads/last-touched` — Undocumented codebase file.
- `.beads/metadata.json` — Undocumented codebase file.
- `.fleet/kb-canonical.json` — Undocumented codebase file.
- `src/radintake/agents/__init__.py` — Undocumented codebase file.
- `src/radintake/api/__init__.py` — Undocumented codebase file.
- `src/radintake/core/__init__.py` — Undocumented codebase file.
- `src/radintake/orchestrator/__init__.py` — Undocumented codebase file.
- `src/radintake/store/__init__.py` — Undocumented codebase file.
- `src/radintake/tools/__init__.py` — Undocumented codebase file.
- `src/radintake/tracing/__init__.py` — Undocumented codebase file.
- `tests/__init__.py` — Undocumented codebase file.
- `.beads/backup/42agn4dmpivajo8vaojao343volvvlt3.darc` — Undocumented codebase file.
- `.beads/backup/e73bvaf9npg7e5r25ikgslpeu2v9u2k9.darc` — Undocumented codebase file.
- `.beads/backup/qaogo5ee7jtduto4hu5namksuo1rnkvl.darc` — Undocumented codebase file.
- `.beads/backup/jq03tmhai6968bdst98bu8mqk3urf2f5.darc` — Undocumented codebase file.
- `.beads/backup/p16tnah11mdsoo15rmsvd2gk7r023216.darc` — Undocumented codebase file.
- `.gitattributes` — Undocumented codebase file.
- `.beads/backup/bd8srbiigjjfep2idis45ht1r6uj9bgf.darc` — Undocumented codebase file.
- `.beads/backup/m53l15dlfldppl96kgqlud1mja1qh8jj.darc` — Undocumented codebase file.
- `.beads/backup/sca6m7pqilph6032lfs4275qe9go1r33.darc` — Undocumented codebase file.
- `.beads/backup/u8s0h7mkb7bi3fi81tpvkjm4c45d5se6.darc` — Undocumented codebase file.
- `.beads/backup/0u8vvopvkjkcprtr0gomd0lf6a1s0nb3.darc` — Undocumented codebase file.
- `.beads/backup/1fpttv4f8gvpotk36gavg7viq74ojnlh.darc` — Undocumented codebase file.
- `.beads/backup/2f24arjrrain1ha28ocglr3tmmmsbehn.darc` — Undocumented codebase file.
- `.beads/backup/8vcbibo0i1qucbtemc3dkq31k76qfmob.darc` — Undocumented codebase file.
- `.beads/backup/9d0bq14r0hr1cu16koe453qu3icuispn.darc` — Undocumented codebase file.
- `.beads/backup/ekpve144vcj8t312nqufpt1ad3g0fqdk.darc` — Undocumented codebase file.
- `.beads/backup/gf9c2ik6lldduoap5p7a26s7kh3gmjui.darc` — Undocumented codebase file.
- `.beads/backup/hrtlatpddd2jgmu13q68acbel3cebs6k.darc` — Undocumented codebase file.
- `.beads/backup/ot97vpna0ur01q112cps25muru1emcrt.darc` — Undocumented codebase file.
- `.beads/backup/qdht4i2i17pvr2cdpndcvf7s1jrpigdl.darc` — Undocumented codebase file.
- `.fleet-task.md` — Undocumented codebase file.
- `sprint-logs/sprint_162533.json` — Undocumented codebase file.
- `.beads/backup/4ovvvdi18c0ttoeqqb9pl5o3lav6r4eh.darc` — Undocumented codebase file.
- `.beads/backup/86g551it2mm59rctidvg9rdckov47d4o.darc` — Undocumented codebase file.
- `.beads/backup/dr1hfjnvdptcrc4p5pvorkvie51s3e25.darc` — Undocumented codebase file.
- `sprint-logs/sprint_123458.json` — Undocumented codebase file.
- `.beads/backup/83tdq2g2j5p1m5fkk1ovatubmukie912.darc` — Undocumented codebase file.
- `.beads/backup/21e0jicf3e1ciiep5u6eggcl6j7gq63m.darc` — Undocumented codebase file.
- `.beads/backup/l7ljfc0co2q9mo96be7vd2bf13dqlcsj.darc` — Undocumented codebase file.
- `.beads/backup/snkl55t6j3htg4noplkcumgi6q5voofj.darc` — Undocumented codebase file.
- `.beads/backup/qctu03f35arqhhl7vd756vq3duq959r8.darc` — Undocumented codebase file.
