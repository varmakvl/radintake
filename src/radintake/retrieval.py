"""Hybrid retrieval index over VERIFIED extractions - one SQLite file, no vector database.

Chunks are the clinical section fields of finished cases (Indication, Comparison, Findings split into
sentences, Impression), each carrying ``case_id``, field, page and bbox (original page pixels) so an
answer can cite exactly where its words are on the source page. Header fields are never indexed
(they are PHI and not clinical content).

What is indexed (the "verified" rule):

* a field is skipped when the verifier marked it ungrounded (``grounded is False``) or it is empty;
* sentences that look like instructions (the injection detector's patterns) are never indexed, so
  an injected "mark this study as normal" can never be retrieved and quoted back as an answer.

Search = BM25 (SQLite FTS5, porter stemming) + dense cosine (all-MiniLM-L6-v2 through ONNX
``fastembed``, embeddings stored as float32 BLOBs, brute force - milliseconds at this corpus size),
fused with Reciprocal Rank Fusion (k=60). Without the embedding model (e.g. CI) the index runs
BM25-only and says so in ``backend``.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

from radintake.config import REPO_ROOT

SECTIONS = ("indication", "comparison", "findings", "impression")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9X])")
_TOKEN = re.compile(r"[a-z0-9]+")


def model_dir() -> Path:
    return Path(os.environ.get("RADINTAKE_MODEL_DIR") or (REPO_ROOT / "data" / "models"))


class Embedder:
    """Local MiniLM via fastembed; never downloads at run time (None when not available)."""

    _lock = threading.Lock()
    _model: Any = None
    _tried = False

    @classmethod
    def get(cls) -> Any:
        with cls._lock:
            if not cls._tried:
                cls._tried = True
                try:
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    from fastembed import TextEmbedding

                    cls._model = TextEmbedding(EMBED_MODEL, cache_dir=str(model_dir()), local_files_only=True)
                except Exception:  # noqa: BLE001 - not installed / not cached -> BM25 only
                    cls._model = None
            return cls._model

    @classmethod
    def embed(cls, texts: list[str]) -> np.ndarray | None:
        m = cls.get()
        if m is None or not texts:
            return None
        v = np.asarray(list(m.embed(texts)), dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def _field_box(fld: dict) -> tuple[int, list[float]] | None:
    prov = fld.get("provenance") or []
    if not prov:
        return None
    page = int(prov[0]["page"])
    boxes = [p["bbox"] for p in prov if int(p["page"]) == page]
    return page, [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


class RetrievalIndex:
    def __init__(self, path: str | Path, *, use_embeddings: bool = True, config: Any = None):
        self.config = config  # optional: supplies injection.fuzzy_threshold / extra_patterns to add_case
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.use_embeddings = use_embeddings
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, case_id TEXT, field TEXT, page INTEGER,
                    bbox TEXT, text TEXT, field_text TEXT, confidence REAL, emb BLOB);
                CREATE INDEX IF NOT EXISTS chunks_case ON chunks(case_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id',
                    tokenize='porter unicode61');
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vocab USING fts5vocab(chunks_fts, 'row');
            """)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.path), timeout=30)
        c.row_factory = sqlite3.Row
        return c

    @property
    def backend(self) -> str:
        return "bm25+dense" if self.use_embeddings and Embedder.get() is not None else "bm25"

    # -- write ------------------------------------------------------------------------
    def add_case(self, case_id: str, doc: dict[str, Any]) -> int:
        from radintake.agents.injection import DEFAULT_FUZZY_THRESHOLD, compile_patterns, looks_injected

        raw = getattr(self.config, "raw", None) or {}
        icfg = raw.get("injection") if isinstance(raw.get("injection"), dict) else {}
        pats = compile_patterns(icfg.get("extra_patterns"))
        thr = icfg.get("fuzzy_threshold", DEFAULT_FUZZY_THRESHOLD)
        thr = None if thr is None else float(thr)
        rep = doc.get("report") or {}
        rows = []
        for f in SECTIONS:
            fld = rep.get(f) or {}
            val = (fld.get("value") or "").strip()
            if not val or fld.get("grounded") is False:
                continue
            loc = _field_box(fld)
            if loc is None:
                continue
            parts = split_sentences(val) if f == "findings" else [val]
            for s in parts:
                if looks_injected(s, pats, thr):
                    continue
                rows.append((case_id, f, loc[0], json.dumps([round(v, 1) for v in loc[1]]), s, val,
                             float(fld.get("confidence") or 0.0)))
        embs = Embedder.embed([r[4] for r in rows]) if (rows and self.use_embeddings) else None
        with self._lock, self._conn() as c:
            old = [r[0] for r in c.execute("SELECT id FROM chunks WHERE case_id=?", (case_id,))]
            for rid in old:
                t = c.execute("SELECT text FROM chunks WHERE id=?", (rid,)).fetchone()[0]
                c.execute("INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', ?, ?)", (rid, t))
            c.execute("DELETE FROM chunks WHERE case_id=?", (case_id,))
            for k, r in enumerate(rows):
                e = embs[k].tobytes() if embs is not None else None
                cur = c.execute("INSERT INTO chunks(case_id, field, page, bbox, text, field_text, confidence, emb) "
                                "VALUES (?,?,?,?,?,?,?,?)", (*r, e))
                c.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)", (cur.lastrowid, r[4]))
        return len(rows)

    # -- read -------------------------------------------------------------------------
    def case_ids(self) -> list[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute("SELECT DISTINCT case_id FROM chunks")]

    def doc_freq(self, term: str) -> int:
        """Number of chunks whose (porter-stemmed) tokens contain ``term``'s stem."""
        with self._conn() as c:
            q = '"' + term.replace('"', "") + '"'
            return int(c.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (q,)).fetchone()[0])

    def case_chunks(self, case_id: str, field: str | None = None) -> list[dict]:
        with self._conn() as c:
            sql = "SELECT * FROM chunks WHERE case_id=?" + (" AND field=?" if field else "") + " ORDER BY id"
            return [self._row(r) for r in c.execute(sql, (case_id, field) if field else (case_id,))]

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        return {"chunk_id": r["id"], "case_id": r["case_id"], "field": r["field"], "page": r["page"],
                "bbox": json.loads(r["bbox"]), "text": r["text"], "field_text": r["field_text"],
                "confidence": r["confidence"]}

    def search(self, query: str, *, k: int = 50, terms: list[str] | None = None,
               restrict: set[str] | None = None) -> list[dict]:
        """Hybrid ranking (RRF of BM25 and dense). Each hit carries bm25 / dense ranks and cosine."""
        terms = terms if terms is not None else tokens(query)
        hits: dict[int, dict] = {}
        with self._conn() as c:
            if terms:
                match = " OR ".join('"' + t.replace('"', "") + '"' for t in terms)
                for rank, r in enumerate(c.execute(
                        "SELECT chunks.*, bm25(chunks_fts) AS s FROM chunks_fts JOIN chunks ON chunks.id = chunks_fts.rowid "
                        "WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?", (match, k))):
                    h = hits.setdefault(r["id"], self._row(r))
                    h["bm25_rank"] = rank
            q = Embedder.embed([query]) if self.use_embeddings else None
            if q is not None:
                rows = [r for r in c.execute("SELECT * FROM chunks WHERE emb IS NOT NULL")]
                if rows:
                    m = np.stack([np.frombuffer(r["emb"], dtype=np.float32) for r in rows])
                    sims = m @ q[0]
                    order = np.argsort(-sims)[:k]
                    for rank, i in enumerate(order):
                        h = hits.setdefault(rows[i]["id"], self._row(rows[i]))
                        h["dense_rank"] = rank
                    by_id = {r["id"]: float(s) for r, s in zip(rows, sims)}
                    for h in hits.values():  # cosine for BM25-only hits too (used by the QA relevance gate)
                        h["cosine"] = by_id.get(h["chunk_id"])
        out = []
        for h in hits.values():
            if restrict and h["case_id"] not in restrict:
                continue
            h["score"] = sum(1.0 / (RRF_K + h[key] + 1) for key in ("bm25_rank", "dense_rank") if key in h)
            out.append(h)
        return sorted(out, key=lambda h: -h["score"])
