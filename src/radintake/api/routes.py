"""HTTP routes: cases, review queue, query, health. Logs carry ids/counts only."""

from __future__ import annotations

import importlib
import shutil
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from radintake.api.app import ApiError, problem
from radintake.logging import get_logger
from radintake.store import review_queue
from radintake.store.cases import CaseStore

router = APIRouter()
log = get_logger("radintake.api.routes")

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_ZIP_MEMBERS = 500
MIN_FREE_BYTES = 100 * 1024 * 1024


def _safe_name(name: str | None) -> str:
    n = PurePosixPath((name or "").replace("\\", "/")).name
    return n if n and n not in (".", "..") else "upload"


def _process(app: Any, case_id: str, path: Path) -> None:
    from radintake.pipeline import process_file

    store: CaseStore = app.state.store
    cfg = app.state.config
    try:
        store.set_running(case_id)
        doc = process_file(path, path.parent / "out", config=cfg, input_name=path.name)
        if doc.get("status") != "error" and doc.get("report"):
            n = app.state.index.add_case(case_id, doc)  # verified section fields only
            log.info("case_indexed", case_id=case_id, chunks=n)
        store.finish(case_id, doc)
        log.info("case_finished", case_id=case_id, status=doc["status"], trace_id=doc["trace_id"])
    except Exception as exc:  # noqa: BLE001 - a case failure never kills the worker
        log.error("case_crashed", case_id=case_id, error_type=type(exc).__name__)
        store.fail(case_id, f"pipeline crashed: {type(exc).__name__}")


def _index_dir(app: Any) -> Path:
    return Path(app.state.config.data_dir) / "cases" / "_index"


def _remember_out_dir(app: Any, case_id: str, out_dir: Path) -> None:
    """case_id -> output dir, so the UI can serve the case's PHI-masked page images."""
    d = _index_dir(app)
    d.mkdir(parents=True, exist_ok=True)
    (d / case_id).write_text(str(out_dir), encoding="utf-8")


def _submit(app: Any, name: str, src: Path) -> str:
    """Register a case for ``src`` (already on disk under the case dir) and queue it."""
    store: CaseStore = app.state.store
    case_id = store.create(name)
    _remember_out_dir(app, case_id, src.parent / "out")
    app.state.pool.submit(_process, app, case_id, src)
    return case_id


async def _save_upload(up: UploadFile, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with dest.open("wb") as fh:
        while chunk := await up.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                fh.close()
                dest.unlink(missing_ok=True)
                raise ApiError(413, "Upload too large", f"limit is {MAX_UPLOAD_BYTES} bytes")
            fh.write(chunk)
    return size


def _unzip(archive: Path, into: Path) -> list[Path]:
    """Flat, bounded extraction: only regular file members, base names only (no path traversal)."""
    out: list[Path] = []
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            if len(infos) > MAX_ZIP_MEMBERS or sum(i.file_size for i in infos) > MAX_UPLOAD_BYTES * 2:
                raise ApiError(413, "Zip bundle too large")
            for i, info in enumerate(infos):
                name = _safe_name(info.filename)
                d = into / f"{i:04d}" / name
                d.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, d.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                out.append(d)
    except zipfile.BadZipFile as exc:
        raise ApiError(400, "Bad zip bundle", "not a readable zip archive") from exc
    return out


@router.post("/cases", status_code=202)
async def post_case(request: Request, file: UploadFile) -> JSONResponse:
    app = request.app
    name = _safe_name(file.filename)
    base = Path(app.state.config.data_dir) / "cases" / uuid.uuid4().hex
    src = base / name
    await _save_upload(file, src)
    if name.lower().endswith(".zip"):
        members = _unzip(src, base / "members")
        if not members:
            raise ApiError(400, "Empty zip bundle")
        ids = [_submit(app, m.name, m) for m in members]
        log.info("bundle_accepted", n_cases=len(ids))
        return JSONResponse({"case_id": ids[0], "case_ids": ids}, status_code=202)
    case_id = _submit(app, name, src)
    log.info("case_accepted", case_id=case_id)
    return JSONResponse({"case_id": case_id, "case_ids": [case_id]}, status_code=202)


@router.get("/cases/{case_id}")
def get_case(case_id: str, request: Request) -> dict[str, Any]:
    c = request.app.state.store.get(case_id)
    if c is None:
        raise ApiError(404, "Case not found", f"no case {case_id}")
    return {"case_id": c["case_id"], "input": c["input"], "status": c["status"], "result": c["result"],
            "flags": c["flags"], "trace_id": c["trace_id"], "error": c["error"]}


@router.get("/cases/{case_id}/pages/{page}.png")
def get_case_page(case_id: str, page: int, request: Request) -> FileResponse:
    """The PHI-MASKED page image the PHI agent produced (never the original upload)."""
    app = request.app
    c = app.state.store.get(case_id)  # also validates case_id before it touches the filesystem
    if c is None:
        raise ApiError(404, "Case not found", f"no case {case_id}")
    marker = _index_dir(app) / case_id
    arts = ((c.get("result") or {}).get("phi") or {}).get("masked_artifacts") or []
    rel = next((a for a in arts if a.endswith(f"/page_{page}.png")), None)
    if rel is None or not marker.is_file():
        raise ApiError(404, "Page image not found", f"no masked image for page {page}")
    out_dir = Path(marker.read_text(encoding="utf-8")).resolve()
    f = (out_dir / rel).resolve()
    if out_dir not in f.parents or not f.is_file():
        raise ApiError(404, "Page image not found", f"no masked image for page {page}")
    return FileResponse(f, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.get("/review-queue")
def get_review_queue(request: Request) -> dict[str, Any]:
    """Persisted cases in ``needs_review``, enriched with queue reasons/fields (ids and names only)."""
    app = request.app
    by_trace = {i["trace_id"]: i for i in review_queue.list_items(app.state.config.data_dir) if i["trace_id"]}
    items = []
    for c in app.state.store.list("needs_review"):
        q = by_trace.get(c["trace_id"], {})
        items.append({"case_id": c["case_id"], "input": c["input"], "trace_id": c["trace_id"],
                      "flags": c["flags"], "reasons": q.get("reasons") or c["flags"],
                      "fields": q.get("fields") or [], "created_at": c["created_at"]})
    return {"items": items}


class QueryBody(BaseModel):
    question: str


@router.post("/query")
def post_query(body: QueryBody, request: Request) -> Any:
    try:
        qa = importlib.import_module("radintake.agents.qa")
        answer = qa.answer
    except (ImportError, AttributeError):
        return problem(503, "Query service unavailable", "the QA agent is not installed")
    res = answer(body.question, request.app.state.index)
    log.info("query_answered", abstained=res["abstained"], n_citations=len(res["citations"]),
             backend=res["retrieval"]["backend"])  # ids/counts only: no question or answer text
    return res


@router.get("/health")
def health(request: Request) -> JSONResponse:
    app = request.app
    cfg = app.state.config
    checks: dict[str, str] = {}
    try:
        checks["db"] = "ok" if app.state.store.ping() else "fail"
    except Exception:  # noqa: BLE001
        checks["db"] = "fail"
    checks["models"] = "ok" if cfg.models and all(m.id for m in cfg.models.values()) else "fail"
    try:
        free = shutil.disk_usage(cfg.data_dir).free
        checks["disk"] = "ok" if free >= MIN_FREE_BYTES else "low"
    except OSError:
        checks["disk"] = "fail"
    ok = all(v == "ok" for v in checks.values())
    return JSONResponse({"status": "ok" if ok else "degraded", "checks": checks, "offline": cfg.offline},
                        status_code=200 if ok else 503)
