"""FastAPI app factory: ``create_app()`` builds the service from the environment config
(``RADINTAKE_DATA_DIR``, ``RADINTAKE_OFFLINE``, ...). Errors are ``application/problem+json``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from radintake.config import Config, load_config
from radintake.logging import configure_logging, get_logger
from radintake.store.cases import CaseStore

PROBLEM = "application/problem+json"
log = get_logger("radintake.api")


def problem(status: int, title: str, detail: str | None = None, **extra: Any) -> JSONResponse:
    body = {"type": "about:blank", "title": title, "status": status, **extra}
    if detail:
        body["detail"] = detail
    return JSONResponse(body, status_code=status, media_type=PROBLEM)


class ApiError(Exception):
    def __init__(self, status: int, title: str, detail: str | None = None):
        self.status, self.title, self.detail = status, title, detail


def create_app(config: Config | None = None, *, max_workers: int = 2) -> FastAPI:
    cfg = config or load_config()
    configure_logging(cfg.log_level)
    store = CaseStore(cfg.data_dir)
    from radintake.retrieval import RetrievalIndex

    index = RetrievalIndex(Path(cfg.data_dir) / "retrieval.sqlite")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        n = store.interrupt_unfinished()
        if n:
            log.warning("cases_interrupted", count=n)
        if not index.case_ids():  # rebuild the retrieval index from finished cases (e.g. after an upgrade)
            for c in store.list():
                if c.get("result") and c["status"] in ("ok", "needs_review"):
                    index.add_case(c["case_id"], c["result"])
        app.state.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="radintake-case")
        try:
            yield
        finally:
            app.state.pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="radintake", version="0.1.0", lifespan=lifespan)
    app.state.config, app.state.store, app.state.index = cfg, store, index

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return problem(exc.status, exc.title, exc.detail)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return problem(exc.status_code, str(exc.detail) if exc.detail else "HTTP error")

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        locs = [".".join(str(x) for x in e.get("loc", ())) for e in exc.errors()]
        return problem(422, "Invalid request", f"invalid fields: {', '.join(locs)}")

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_error", error_type=type(exc).__name__)
        return problem(500, "Internal server error", type(exc).__name__)

    from radintake.api.routes import router

    app.include_router(router)
    return app
