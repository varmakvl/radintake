---
type: Subsystem
title: Retrieval / QA Agent
description: Hybrid BM25 + dense retrieval over verified extractions in one SQLite file (no vector database), and an extractive QA agent whose answers are verbatim, verified quotes cited with case, page and box - with negation handling and abstention.
resource: src/radintake/agents/qa.py
tags: [retrieval, qa, rag, citations, abstention, negation, prompt-injection, sqlite, fts5]
status: stable
generated:
  by: Claude/opus-5.5
  at: 2026-11-26T16:36:54.984Z
sources:
  - src/radintake/retrieval.py
  - src/radintake/agents/qa.py
  - src/radintake/api/routes.py
  - src/radintake/api/app.py
  - src/radintake/eval/golden_qa.py
  - eval/dev_qa.jsonl
  - tests/unit/test_retrieval_qa.py
  - tests/integration/test_api_e2e.py
  - Dockerfile
---

# Index (retrieval.py)

Chunks = section fields of finished cases (Indication, Comparison, Findings split into sentences,
Impression) with `case_id`, field, page and bbox (original page pixels). Header fields are never
indexed (PHI, not clinical content). **Verified only:** a field the verifier marked ungrounded is
skipped, and any sentence matching the injection patterns is never indexed - so injected text can't
be retrieved or quoted back. Search = SQLite FTS5 BM25 (porter stemming) + dense cosine with
all-MiniLM-L6-v2 through ONNX `fastembed` (no torch), fused by Reciprocal Rank Fusion (k=60). Model
files live in `data/models` (dev) / `/models` (Docker, fetched at build time); never downloaded at run
time. Without them the index runs BM25-only and reports `backend: "bm25"` (CI).

# Decision: no vector database (2026-11-24)

A few hundred reports = a few thousand chunks; brute-force cosine over float32 BLOBs takes
milliseconds. A vector DB would add a service and a failure mode for no gain. Revisit past ~10^6
chunks (`sqlite-vec` or a dedicated store).

# QA agent (agents/qa.py)

- **Extractive, no generator.** Every answer line is a verbatim quote of an indexed field, cited
  `{case_id, field, page, bbox, quote}`; each quote is re-verified against the stored field text
  before it is returned. Document text never acts as an instruction, and nothing can be hallucinated.
  (The local `qwen2.5:3b` could add a one-line summary over verified quotes; not enabled - it would
  only add risk to a metric that is already 1.0 on citations.)
- **Negation (NegEx-style).** A finding term counts only if no cue (no, not, without, negative, free,
  resolved, clear...) precedes it in the same clause - "No pneumonia" is not a pneumonia report.
- **Abstention.** A content term absent from the corpus that is a head concept (not a modifier such
  as "left", "indwelling") -> `concept_not_in_corpus` unless the dense match is very close (cos >= 0.75);
  no chunk with >= 50% affirmed coverage -> `insufficient_evidence`.
- **Study-level answers.** Best-covered studies first; each gets its evidence sentence plus its
  impression. A study id named in the question answers from that study directly.

API: `POST /query` -> `{answer, abstained, reason, studies, citations, retrieval:{backend, n_hits}}`;
cases are indexed when they finish; the index is rebuilt from the case store at startup if empty.
Logs carry counts only (no question or answer text).

# Dev results (eval/dev_qa.jsonl, 39 questions, OCR-first dev outputs)

| Step | recall@5 | answer correctness | correct abstention | citation accuracy |
|---|---|---|---|---|
| first version | 0.47 | 0.41 | 0.72 | 1.00 |
| + named-study path, negation, modifier-aware abstention, best-coverage selection | 0.58 | 0.77 | 0.97 | 1.00 |

Unanswerable questions abstained: 5/5; false abstention on answerable: 3%; injection probes correct:
0.73 (misses are OCR noise in the impression, not retrieval). Golden QA (eval/golden_qa.jsonl) is
scored only in the final run ([Evaluation Harness](/operations/evaluation-harness.md)).

Related: [Verifier](/subsystems/verifier-agent.md), [API Service](/subsystems/api-service.md).

# Guardrails & Anti-Regression Checklist

- Never index header fields or untrusted / injected sentences.
- Never return a citation whose quote does not verify against the stored field text.
- Tune on `eval/dev_qa.jsonl` only; the golden QA set is for the single final run.
