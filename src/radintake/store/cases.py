"""SQLite case store for the API (``<RADINTAKE_DATA_DIR>/cases.sqlite``).

One row per uploaded input: ``case_id`` (uuid), input name, status
(``pending|running|ok|needs_review|error``), flags, trace_id and the pipeline result document.
Survives restarts. Rows are only ever inserted or advanced - never deleted from here.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DB_NAME = "cases.sqlite"
TERMINAL = ("ok", "needs_review", "error")
_SCHEMA = """CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    input_name TEXT NOT NULL,
    status TEXT NOT NULL,
    flags TEXT NOT NULL DEFAULT '[]',
    trace_id TEXT,
    result TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class CaseStore:
    def __init__(self, data_dir: str | Path):
        self.path = Path(data_dir) / DB_NAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as con:
            con.execute(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path), timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def _run(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            con = self._conn()
            try:
                with con:
                    return con.execute(sql, args).fetchall()
            finally:
                con.close()

    def ping(self) -> bool:
        return self._run("SELECT 1")[0][0] == 1

    def create(self, input_name: str) -> str:
        cid = uuid.uuid4().hex
        now = _now()
        self._run("INSERT INTO cases (case_id, input_name, status, created_at, updated_at) VALUES (?,?,?,?,?)",
                  (cid, input_name, "pending", now, now))
        return cid

    def set_running(self, case_id: str) -> None:
        self._run("UPDATE cases SET status='running', updated_at=? WHERE case_id=?", (_now(), case_id))

    def finish(self, case_id: str, doc: dict[str, Any]) -> None:
        self._run("UPDATE cases SET status=?, flags=?, trace_id=?, result=?, error=?, updated_at=? "
                  "WHERE case_id=?",
                  (doc["status"], json.dumps(doc.get("flags") or []), doc.get("trace_id"),
                   json.dumps(doc, ensure_ascii=False), doc.get("error"), _now(), case_id))

    def fail(self, case_id: str, error: str) -> None:
        self._run("UPDATE cases SET status='error', error=?, updated_at=? WHERE case_id=?",
                  (error, _now(), case_id))

    def interrupt_unfinished(self) -> int:
        """Cases left pending/running by a previous process cannot continue: mark them errors."""
        rows = self._run("UPDATE cases SET status='error', error='interrupted by service restart', "
                         "updated_at=? WHERE status IN ('pending','running') RETURNING case_id", (_now(),))
        return len(rows)

    @staticmethod
    def _row(r: sqlite3.Row) -> dict[str, Any]:
        return {"case_id": r["case_id"], "input": r["input_name"], "status": r["status"],
                "flags": json.loads(r["flags"]), "trace_id": r["trace_id"],
                "result": json.loads(r["result"]) if r["result"] else None, "error": r["error"],
                "created_at": r["created_at"], "updated_at": r["updated_at"]}

    def get(self, case_id: str) -> dict[str, Any] | None:
        rows = self._run("SELECT * FROM cases WHERE case_id=?", (case_id,))
        return self._row(rows[0]) if rows else None

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._run("SELECT * FROM cases WHERE status=? ORDER BY created_at, rowid", (status,))
        else:
            rows = self._run("SELECT * FROM cases ORDER BY created_at, rowid")
        return [self._row(r) for r in rows]
