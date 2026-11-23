"""Append-only human review queue (SQLite under ``RADINTAKE_DATA_DIR``).

Entries hold ids, flags, reason codes and field NAMES only - never PHI or extracted values:
``{id, case_id, trace_id, flags, reasons, fields, created_at}``. The API reads the queue through
:func:`list_items`; batch runs additionally append each entry to ``<out>/review_queue.jsonl``
(:func:`append_jsonl`). The table is never updated or deleted from here (append-only).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_NAME = "review_queue.sqlite"
JSONL_NAME = "review_queue.jsonl"
_lock = threading.Lock()
_SCHEMA = """CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    trace_id TEXT,
    flags TEXT NOT NULL,
    reasons TEXT NOT NULL,
    fields TEXT NOT NULL,
    created_at TEXT NOT NULL
)"""


def _default_dir(data_dir: str | Path | None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    from radintake.config import load_config

    return load_config().data_dir


def db_path(data_dir: str | Path | None = None) -> Path:
    return _default_dir(data_dir) / DB_NAME


def _connect(data_dir: str | Path | None) -> sqlite3.Connection:
    p = db_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), timeout=30)
    con.execute(_SCHEMA)
    return con


def add_item(case_id: str, *, flags: list[str] | None = None, reasons: list[str] | None = None,
             fields: list[str] | None = None, trace_id: str | None = None,
             data_dir: str | Path | None = None, created_at: str | None = None) -> dict[str, Any]:
    """Append one entry and return it. ``case_id`` is the case/input name (an id, not content)."""
    item = {"case_id": case_id, "trace_id": trace_id, "flags": sorted(set(flags or [])),
            "reasons": list(reasons or []), "fields": sorted(set(fields or [])),
            "created_at": created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with _lock:
        con = _connect(data_dir)
        try:
            with con:
                cur = con.execute(
                    "INSERT INTO review_queue (case_id, trace_id, flags, reasons, fields, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (item["case_id"], item["trace_id"], json.dumps(item["flags"]),
                     json.dumps(item["reasons"]), json.dumps(item["fields"]), item["created_at"]))
            item["id"] = cur.lastrowid
        finally:
            con.close()
    return item


def list_items(data_dir: str | Path | None = None, *, case_id: str | None = None) -> list[dict[str, Any]]:
    """All entries, oldest first (optionally only one ``case_id``)."""
    if not db_path(data_dir).exists():
        return []
    con = _connect(data_dir)
    try:
        q = "SELECT id, case_id, trace_id, flags, reasons, fields, created_at FROM review_queue"
        args: tuple = ()
        if case_id is not None:
            q, args = q + " WHERE case_id = ?", (case_id,)
        rows = con.execute(q + " ORDER BY id", args).fetchall()
    finally:
        con.close()
    return [{"id": r[0], "case_id": r[1], "trace_id": r[2], "flags": json.loads(r[3]),
             "reasons": json.loads(r[4]), "fields": json.loads(r[5]), "created_at": r[6]} for r in rows]


def append_jsonl(out_dir: str | Path, item: dict[str, Any]) -> Path:
    """Append the entry to ``<out_dir>/review_queue.jsonl`` (batch runs)."""
    p = Path(out_dir) / JSONL_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock, p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return p
