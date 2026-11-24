---
type: Subsystem
title: Retrieval / QA Agent
description: Planned hybrid keyword + embedding retrieval over verified extractions, all inside SQLite (no vector database), with extractive answers that cite case, page and box, quote verification, abstention on weak evidence and untrusted-data spotlighting.
resource: src/radintake/api/routes.py
tags: [retrieval, qa, rag, citations, abstention, prompt-injection, planned]
status: draft
generated:
  by: Claude/opus-5.5
  at: 2026-11-24T10:26:13.955Z
sources:
  - PLAN.md
  - requirements.md
  - tests/integration/test_api_e2e.py
  - configs/models.yaml
---

# Status

**Not built.** `POST /query` exists as a route in `api/routes.py` but has no retrieval behind it.
Beads: `rad-beb.9.1` (index), `rad-beb.9.2` (QA agent), `rad-beb.9.3` (test).

# Intended design (PLAN.md P4)

- Index only VERIFIED fields (review-pending ones excluded or flagged), chunked per field, findings
  split by sentence, each chunk carrying case id, page and bbox.
- Hybrid retrieval: SQLite FTS5 BM25 + dense embeddings (`embedder` in `configs/models.yaml`:
  `all-MiniLM-L6-v2`, local) fused with Reciprocal Rank Fusion; optional MeSH-aware query expansion.
- Answers are extractive first (the cited sentences themselves); `llm_answer` (Qwen2.5-3B in
  Ollama) may add a one-line summary over the verified quotes only.
- Answerer returns `{answer, citations:[{case_id, page, bbox, quote}], abstained}`. Document text is
  wrapped as delimited untrusted data; every citation quote is checked against its chunk, unverifiable
  claims are dropped, and the agent abstains when retrieval scores fall below a tuned threshold or no
  citation survives.
- Offline mode must still answer (extractive answer from top chunks) so the integration test runs in CI.

# Contract it must satisfy

`tests/integration/test_api_e2e.py`: the granuloma question cites the clean report on page 1; the
femur question abstains; the injected study still answers "pneumonia". Metrics (recall@5, citation
accuracy, answer correctness, correct abstention) come from the [Evaluation Harness](/operations/evaluation-harness.md).

# Decision: no vector database (2026-11-24)

Retrieval is required by the brief (Priority 4, graded on recall@5, citation accuracy, answer
correctness and correct abstention), but the corpus is a few hundred reports — a few thousand
chunks. Embeddings are stored as BLOBs in the same SQLite file as the FTS5 index and compared by
brute-force cosine in numpy (milliseconds at this size). A separate vector database would add a
service, a container and a failure mode for no measurable gain. Revisit only past ~10^6 chunks
(then `sqlite-vec` or a dedicated store).

Related: [Verifier](/subsystems/verifier-agent.md), [API Service](/subsystems/api-service.md),
[Guarded Model Client](/subsystems/model-client-guard.md).
