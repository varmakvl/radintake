"""Retrieval / QA agent: extractive, cited answers over verified extractions, with abstention.

``answer(question, index)`` returns::

    {"answer": str, "abstained": bool, "reason": str | None,
     "studies": [case_id, ...],                       # ranked, at most max_studies
     "citations": [{"case_id", "field", "page", "bbox", "quote"}, ...],
     "retrieval": {"backend": "bm25+dense" | "bm25", "n_hits": int}}

Design choices (and why):

* **Extractive, no generator by default.** Every sentence of an answer is a verbatim quote of an
  indexed field, cited with case, page and box. Nothing in a document is ever executed or obeyed:
  document text only flows into the answer as a quoted, cited string, and instruction-like sentences
  are not even indexed (``radintake.retrieval``). This makes prompt injection through report text
  structurally unable to change the answer, and hallucination impossible by construction.
* **Quote verification.** Each citation's quote is re-checked against the stored field text of its
  case before it is returned; a quote that does not verify is dropped.
* **Abstention.** (1) a content term of the question that occurs nowhere in the corpus (nor any of
  its listed synonyms) and no strong dense match -> ``concept_not_in_corpus``; (2) no chunk passing
  the relevance gate -> ``insufficient_evidence``. The gate needs lexical coverage of the question's
  corpus-known content terms (>= 0.5) or a high dense cosine (>= 0.62).
* **Study-level answers.** Questions like "Which studies ...? Give the impression for each." are
  answered per study: the evidence sentence plus the study's impression, both cited.
"""

from __future__ import annotations

import re
from typing import Any

from radintake.retrieval import RetrievalIndex, tokens

STOP = frozenset({
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "by", "can", "case", "cases", "could",
    "describe", "described", "describes", "did", "do", "does", "each", "evidence", "exam", "exams",
    "find", "finding", "findings", "for", "found", "from", "give", "has", "have", "how", "image",
    "images", "impression", "impressions", "in", "is", "it", "its", "larger", "list", "mention",
    "mentioned", "mentions", "note", "noted", "notes", "of", "on", "or", "patient", "patients",
    "please", "read", "report", "reported", "reports", "show", "showed", "shows", "studies", "study",
    "such", "than", "that", "the", "their", "them", "there", "these", "this", "those", "to", "was",
    "were", "what", "which", "who", "with"
})
SYNONYMS: dict[str, list[str]] = {
    "cardiomegaly": ["enlarged", "enlargement", "cardiomegaly"],
    "calcinosis": ["calcified", "calcification", "calcifications"],
    "cicatrix": ["scar", "scarring", "scarred"],
    "atelectasis": ["atelectatic", "atelectasis"],
    "edema": ["edema", "congestion", "vascular"],
    "congestion": ["edema", "congestion"],
    "opacity": ["opacity", "opacities", "opacification", "density", "densities"],
    "airspace": ["airspace", "consolidation", "opacity"],
    "catheters": ["catheter", "picc", "line", "port", "tube"],
    "catheter": ["catheter", "picc", "line", "port", "tube"],
    "instruments": ["clips", "sutures", "suture", "clip", "wires", "sternotomy"],
    "device": ["pacemaker", "device", "port", "stimulator", "defibrillator"],
    "atherosclerosis": ["atherosclerotic", "calcified", "tortuous"],
    "fractures": ["fracture", "fractures", "fractured"],
    "technically": ["limited", "rotated", "underpenetrated", "exposure"],
    "unsatisfactory": ["limited", "rotated", "suboptimal", "poor"],
}
# Modifiers: a question word missing from the corpus that is only a modifier does not make the
# question unanswerable ("indwelling catheters", "implanted medical device"); a missing head concept
# ("femur", "pneumoperitoneum", "medication") does.
MODIFIERS = frozenset({
    "left", "right", "bilateral", "upper", "lower", "middle", "small", "large", "mild", "moderate", "severe",
    "acute", "chronic", "sided", "indwelling", "implanted", "medical", "technically", "unsatisfactory",
    "quality", "changes", "degenerative", "deformity", "free", "air", "under", "lung", "lungs", "thoracic",
    "spine", "such", "clips", "sutures", "scarring", "biopsy", "proven", "cm", "dislocated", "prosthesis",
})
# NegEx-style cues: a finding term preceded (within the clause) by one of these is negated -
# "No pneumothorax or pneumonia" does not report pneumonia.
NEGATION = frozenset({"no", "not", "without", "negative", "free", "absent", "resolved", "resolution",
                      "excluded", "never", "none", "neither", "nor", "clear"})
CLAUSE_BREAK = frozenset({"but", "however", "although", "though", "except", "with"})
STRONG_COSINE = 0.62
MISSING_CONCEPT_COSINE = 0.75  # a head concept absent from the corpus needs a very close dense match
MIN_COVERAGE = 0.5


def content_terms(question: str) -> list[str]:
    seen: list[str] = []
    for t in tokens(question):
        if len(t) >= 3 and t not in STOP and not t.isdigit() and t not in seen:
            seen.append(t)
    return seen


def _stem(t: str) -> str:
    return t[:6] if len(t) > 6 else t


def _affirmed(sentence_tokens: list[str], i: int, window: int = 6) -> bool:
    """False when token ``i`` sits in a negated clause (a cue within ``window`` tokens before it)."""
    for j in range(i - 1, max(-1, i - window - 1), -1):
        t = sentence_tokens[j]
        if t in CLAUSE_BREAK:
            return True
        if t in NEGATION:
            return False
    return True


def _coverage(text: str, terms: list[str]) -> float:
    """Best per-sentence share of ``terms`` present AND affirmed (not negated) in one sentence."""
    if not terms:
        return 0.0
    best = 0.0
    for sent in re.split(r"(?<=[.;])\s+", text):
        toks = tokens(sent)
        stems = [_stem(t) for t in toks]
        hit = 0
        for t in terms:
            alts = {_stem(a) for a in (t, *SYNONYMS.get(t, []))}
            idx = [i for i, s in enumerate(stems) if s in alts]
            if idx and (t in MODIFIERS or any(_affirmed(toks, i) for i in idx)):
                hit += 1
        best = max(best, hit / len(terms))
    return best


def _abstain(reason: str, backend: str) -> dict[str, Any]:
    return {"answer": "I cannot answer this from the indexed reports: no study contains sufficient evidence.",
            "abstained": True, "reason": reason, "studies": [], "citations": [],
            "retrieval": {"backend": backend, "n_hits": 0}}


def _cite(index: RetrievalIndex, ch: dict) -> dict | None:
    """Citation for a chunk, only if its quote verifies against the stored field text of its case."""
    stored = {c["text"] for c in index.case_chunks(ch["case_id"], ch["field"])}
    if ch["text"] not in stored or ch["text"] not in (ch.get("field_text") or ch["text"]):
        return None
    return {"case_id": ch["case_id"], "field": ch["field"], "page": ch["page"], "bbox": ch["bbox"], "quote": ch["text"]}


def answer(question: str, index: RetrievalIndex, *, max_studies: int = 10) -> dict[str, Any]:
    backend = index.backend
    q = (question or "").strip()
    if not q:
        return _abstain("empty_question", backend)
    known = index.case_ids()
    named = {cid for cid in known if re.search(rf"\b{re.escape(cid)}\b", q, re.IGNORECASE)}
    terms = content_terms(q)
    if named:  # "What is the impression for study D0077?" - the study id is not a content term
        terms = [t for t in terms if not any(t == cid.lower() for cid in named)]
    if named:  # an explicitly named study: answer from that study's own fields, no ranking needed
        hits = [{**ch, "score": 1.0, "coverage": 1.0} for cid in sorted(named)
                for ch in index.case_chunks(cid, "impression") or index.case_chunks(cid)]
    else:
        hits = index.search(q, terms=[t for t in terms for t in (t, *SYNONYMS.get(t, []))] or None)
    best_cos = max((h.get("cosine") or 0.0 for h in hits), default=0.0)
    in_corpus = [t for t in terms if index.doc_freq(t) > 0 or any(index.doc_freq(s) > 0 for s in SYNONYMS.get(t, []))]
    missing = [t for t in terms if t not in in_corpus and t not in MODIFIERS]
    if not named and missing and best_cos < MISSING_CONCEPT_COSINE:
        return _abstain("concept_not_in_corpus", backend)
    heads = [t for t in in_corpus if t not in MODIFIERS] or in_corpus
    for h in hits:
        if "coverage" not in h:
            h["coverage"] = _coverage(h["text"], in_corpus)
            h["head_coverage"] = _coverage(h["text"], heads)
    gated = [h for h in hits if named or (h["coverage"] >= MIN_COVERAGE and h.get("head_coverage", 1.0) > 0)]
    if not gated:
        return _abstain("insufficient_evidence", backend)
    # keep the best-supported studies: highest affirmed coverage first, fused retrieval score second
    top_cov = max(h["coverage"] for h in gated)
    gated = sorted((h for h in gated if h["coverage"] >= top_cov - 1e-9),
                   key=lambda h: -h["score"]) + sorted((h for h in gated if h["coverage"] < top_cov - 1e-9),
                                                       key=lambda h: (-h["coverage"], -h["score"]))
    keep_n = max_studies if top_cov < 1.0 else min(max_studies, len({h["case_id"] for h in gated
                                                                     if h["coverage"] >= top_cov - 1e-9}))

    studies: list[str] = []
    evidence: dict[str, dict] = {}
    for h in gated:  # already ranked by fused score
        if h["case_id"] not in evidence:
            evidence[h["case_id"]] = h
            studies.append(h["case_id"])
        if len(studies) >= keep_n:
            break

    lines, citations = [], []
    for cid in studies:
        ev = evidence[cid]
        imp = next(iter(index.case_chunks(cid, "impression")), None)
        parts = []
        for ch in ([ev] if ev["field"] == "impression" else [ev, imp]):
            if ch is None:
                continue
            c = _cite(index, ch)
            if c is None:
                continue
            citations.append(c)
            parts.append(f"{ch['field'].capitalize()}: \"{ch['text']}\"")
        if parts:
            lines.append(f"- Study {cid} - " + "; ".join(parts))
    if not citations:
        return _abstain("no_verifiable_citation", backend)
    head = f"{len(lines)} stud{'y' if len(lines) == 1 else 'ies'} matched (quotes are verbatim from the reports):"
    return {"answer": head + "\n" + "\n".join(lines), "abstained": False, "reason": None, "studies": studies,
            "citations": citations, "retrieval": {"backend": backend, "n_hits": len(hits)}}
