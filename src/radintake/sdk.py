"""Public SDK: run the radintake pipeline from Python, no CLI or server needed.

Stability: only the names re-exported in ``radintake.__all__`` are public (semver, see
``docs/api-stability.md``); everything else here (``materialise``, helpers) is internal.

::

    from radintake.sdk import Pipeline, RadintakeConfig

    pipe = Pipeline(RadintakeConfig(offline=True))
    result = pipe.process("report.pdf")          # or bytes, a PIL image, a numpy array,
    print(result.status, result.report.impression.value)   # or a list of page images

Write-free by default
---------------------
With no ``output_dir`` / ``trace_dir`` / ``data_dir``, :meth:`Pipeline.process` leaves nothing
behind: inputs that are not a path are spooled into a private temporary directory, stage
artefacts (PHI-masked page PNGs, the de-identified DICOM), the trace and the review-queue entry
of a ``needs_review`` case are written under that directory too, and the directory is deleted
before ``process`` returns. The masked artefacts come back in memory on the result
(:attr:`Result.masked_pages`, :attr:`Result.dicom_bytes`); ``phi.masked_artifacts`` and
``dicom.output_path`` keep their ``cli_output.v1`` meaning - names relative to ``output_dir`` -
and only exist on disk when an ``output_dir`` was given.

* ``output_dir`` -> ``<output_dir>/<input>.json`` plus the artefacts (exactly what the batch CLI
  writes per file, including ``review_queue.jsonl`` for ``needs_review`` cases);
* ``trace_dir`` -> one trace run per :class:`Pipeline` under ``<trace_dir>/<run_id>/``;
* ``data_dir`` -> the persistent review queue (``review_queue.sqlite``) and model caches.

Configuration
-------------
``Pipeline(config)`` takes a :class:`RadintakeConfig` (code only, no YAML read for the main
config), an internal :class:`radintake.config.Config` (e.g. from ``load_config()``), or ``None``
(``load_config()``: ``configs/*.yaml`` + env). ``RADINTAKE_OFFLINE`` /
``RADINTAKE_FORBID_MODEL_CALLS`` still apply - they can only make a run *more* restricted, and the
model guard re-reads them at call time.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import os
import tempfile
import threading
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from radintake.config import Config, config_from_raw, load_config
from radintake.results import Result

PathLike = str | os.PathLike[str]


class RadintakeError(Exception):
    """Base class of the public radintake SDK exceptions."""


class UnsupportedInputError(RadintakeError, TypeError):
    """``source`` is not a path, bytes, a PIL image, a numpy array or a list of page images."""


class ConfigError(RadintakeError, TypeError):
    """``config`` is not a :class:`RadintakeConfig`, a ``radintake.config.Config`` or ``None``."""

_SUFFIX = {"dicom": ".dcm", "pdf": ".pdf", "tiff": ".tiff"}
_PIL_SUFFIX = {"PNG": ".png", "JPEG": ".jpg", "BMP": ".bmp", "WEBP": ".webp", "GIF": ".gif",
               "TIFF": ".tiff"}


class BudgetConfig(BaseModel):
    """Planner budgets (``budgets:`` in ``configs/default.yaml``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_steps: int = Field(default=20, ge=1)
    stage_timeout_s: float = Field(default=300.0, gt=0)
    retries: int = Field(default=1, ge=0)
    backoff_s: float = Field(default=0.5, ge=0)
    case_deadline_s: float = Field(default=900.0, gt=0)


class ModelConfig(BaseModel):
    """One model registry entry (``models:`` in ``configs/models.yaml``)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    id: str
    kind: Literal["local", "api"] = "local"


class RadintakeConfig(BaseModel):
    """Explicit, immutable pipeline configuration - YAML optional.

    ``sections`` passes any other config section through verbatim (``preprocess``,
    ``extraction``, ``phi``, ``dicom_deid``, ``stages``, ``loader`` ...), exactly as it would
    appear in ``configs/default.yaml``. The typed fields win over the same keys in ``sections``.
    ``data_dir`` / ``trace_dir`` are the persistence defaults of a :class:`Pipeline` built from
    this config (``None`` = write-free).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    offline: bool = False
    forbid_model_calls: bool = False
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    pdf_dpi: int = Field(default=200, ge=36, le=1200)
    seed: int = 140
    ocr_engine: Literal["tesseract", "paddle"] = "tesseract"
    log_level: str = "INFO"
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    sections: dict[str, Any] = Field(default_factory=dict)
    data_dir: Path | None = None
    trace_dir: Path | None = None

    def to_raw(self) -> dict[str, Any]:
        """The equivalent merged YAML mapping."""
        import copy

        raw: dict[str, Any] = copy.deepcopy(dict(self.sections))
        loader = dict(raw.get("loader") or {})
        loader["pdf_dpi"] = self.pdf_dpi
        ocr = dict(raw.get("ocr") or {})
        ocr["engine"] = self.ocr_engine
        logging_ = dict(raw.get("logging") or {})
        logging_["level"] = self.log_level
        raw.update(
            seed=self.seed,
            runtime={"offline": self.offline, "forbid_model_calls": self.forbid_model_calls},
            budgets=self.budgets.model_dump(),
            loader=loader, ocr=ocr, logging=logging_,
            models={k: v.model_dump() for k, v in self.models.items()},
            paths={"data_dir": str(self.data_dir) if self.data_dir else "data",
                   "trace_dir": str(self.trace_dir) if self.trace_dir else None},
        )
        return raw

    def to_config(self) -> Config:
        """Build the internal :class:`radintake.config.Config` (env flags may only tighten it)."""
        cfg = config_from_raw(self.to_raw())
        return dataclasses.replace(cfg, offline=cfg.offline or self.offline,
                                   forbid_model_calls=cfg.forbid_model_calls or self.forbid_model_calls)


def _to_pil(img: Any) -> Any:
    from PIL import Image

    if isinstance(img, Image.Image):
        return img
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        np = None
    if np is not None and isinstance(img, np.ndarray):
        arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise UnsupportedInputError(f"unsupported page image type: {type(img).__name__}")


def _bytes_suffix(data: bytes) -> str:
    from radintake.orchestrator.router import sniff

    kind = sniff(data[:4096])
    if kind in _SUFFIX:
        return _SUFFIX[kind]
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return _PIL_SUFFIX.get(str(im.format), ".png")
    except Exception:  # noqa: BLE001 - unknown bytes: the router reports them as unknown
        return ".bin"


def _safe_input_name(name: str | None, default: str) -> str:
    n = Path(str(name)).name if name else ""
    return n if n not in ("", ".", "..") else default


def _is_image(obj: Any) -> bool:
    try:
        _to_pil(obj)
        return True
    except TypeError:
        return False


def materialise(source: Any, folder: Path, input_name: str | None = None) -> tuple[Path, str]:
    """Return ``(path, input_name)`` for ``source``; non-path sources are written into ``folder``."""
    if isinstance(source, (str, os.PathLike)):
        p = Path(source)
        return p, _safe_input_name(input_name, p.name or "input")
    folder.mkdir(parents=True, exist_ok=True)
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
        suffix = _bytes_suffix(data)
        name = _safe_input_name(input_name, "input" + suffix)
        dst = folder / (name if Path(name).suffix else name + suffix)
        dst.write_bytes(data)
        return dst, name
    if isinstance(source, Sequence) and not isinstance(source, str):
        pages = [_to_pil(x) for x in source]
        if not pages:
            raise ValueError("an empty page list cannot be processed")
        name = _safe_input_name(input_name, "pages.tiff")
        dst = folder / (name if Path(name).suffix.lower() in (".tif", ".tiff") else name + ".tiff")
        first, rest = pages[0], pages[1:]
        first.save(dst, format="TIFF", save_all=True, append_images=rest)
        return dst, name
    if _is_image(source):
        name = _safe_input_name(input_name, "image.png")
        dst = folder / (name if Path(name).suffix.lower() == ".png" else name + ".png")
        _to_pil(source).save(dst, format="PNG")
        return dst, name
    raise UnsupportedInputError("source must be a path, bytes, a PIL image, a numpy array or a list of page "
                    f"images (got {type(source).__name__})")


class Pipeline:
    """Reusable, thread-safe entry point to the radintake pipeline.

    ``output_dir`` / ``trace_dir`` / ``data_dir`` opt into persistence (see the module doc);
    without them every :meth:`process` call is write-free.
    """

    def __init__(self, config: RadintakeConfig | Config | None = None, *,
                 output_dir: PathLike | None = None, trace_dir: PathLike | None = None,
                 data_dir: PathLike | None = None):
        if isinstance(config, RadintakeConfig):
            self._config = config.to_config()
            trace_dir = trace_dir if trace_dir is not None else config.trace_dir
            data_dir = data_dir if data_dir is not None else config.data_dir
        elif isinstance(config, Config):
            self._config = config
        elif config is None:
            self._config = load_config()
        else:
            raise ConfigError("config must be a RadintakeConfig, a radintake.config.Config or None")
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self._trace_writer: Any = None
        self._lock = threading.Lock()

    @property
    def config(self) -> Config:
        """The effective internal config."""
        return self._config

    # ----------------------------------------------------------------------------- tracing
    def _shared_trace_writer(self) -> Any:
        from radintake.tracing.writer import TraceWriter

        with self._lock:
            if self._trace_writer is None and self.trace_dir is not None:
                self._trace_writer = TraceWriter(self.trace_dir, model_ids=self._config.model_ids,
                                                 offline=self._config.offline)
            return self._trace_writer

    @property
    def run_id(self) -> str | None:
        """Trace run id (only when tracing to a ``trace_dir``; ``None`` otherwise)."""
        tw = self._shared_trace_writer()
        return tw.run_id if tw is not None else None

    def close(self) -> None:
        """Finalise the trace run manifest (no-op when not tracing)."""
        if self._trace_writer is not None:
            self._trace_writer.finalize()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----------------------------------------------------------------------------- process
    def process(self, source: Any, *, input_name: str | None = None,
                output_dir: PathLike | None = None) -> Result:
        """Process one input and return its typed :class:`Result`. Never raises for a bad input
        file (it becomes ``status: "error"``); raises :class:`UnsupportedInputError` (a
        ``TypeError``) for an unsupported ``source``.
        """
        from radintake.pipeline import process_file
        from radintake.tracing.writer import TraceWriter

        out = Path(output_dir) if output_dir is not None else self.output_dir
        with tempfile.TemporaryDirectory(prefix="radintake-", ignore_cleanup_errors=True) as td:
            tmp = Path(td)
            path, name = materialise(source, tmp / "in", input_name)
            work = out if out is not None else tmp / "out"
            cfg = dataclasses.replace(self._config, trace_dir=None,
                                      data_dir=self.data_dir if self.data_dir is not None else tmp / "data")
            tw = self._shared_trace_writer() or TraceWriter(tmp / "traces", model_ids=cfg.model_ids,
                                                            offline=cfg.offline)
            doc = process_file(path, work, config=cfg, trace_writer=tw, input_name=name,
                               write=out is not None)
            if tw is self._trace_writer:
                tw.finalize()
            masked = _read_artifacts(work, (doc.get("phi") or {}).get("masked_artifacts") or [], ".png")
            dicom_path = (doc.get("dicom") or {}).get("output_path")
            dicom_bytes = _read_artifacts(work, [dicom_path], None) if dicom_path else []
        result = Result.model_validate(doc)
        result.masked_pages = masked
        result.dicom_bytes = dicom_bytes[0] if dicom_bytes else None
        return result

    def process_batch(self, sources: Iterable[Any], *, output_dir: PathLike | None = None,
                      on_result: Callable[[Result], None] | None = None) -> list[Result]:
        """Process every source in order; one :class:`Result` per source.

        One bad input never stops the batch: a corrupt file is the usual ``status: "error"``
        result, and an unsupported source or an unexpected crash becomes an ``error`` result too
        (written to ``output_dir`` like any other). ``on_result`` is called after each input.
        """
        results: list[Result] = []
        for i, source in enumerate(sources):
            try:
                r = self.process(source, output_dir=output_dir)
            except Exception as exc:  # noqa: BLE001 - one bad input never stops the batch
                r = self._crash_result(source, i, exc, output_dir)
            results.append(r)
            if on_result is not None:
                on_result(r)
        return results

    async def aprocess(self, source: Any, *, input_name: str | None = None,
                       output_dir: PathLike | None = None) -> Result:
        """:meth:`process` in a worker thread (does not block the event loop)."""
        return await asyncio.to_thread(self.process, source, input_name=input_name,
                                       output_dir=output_dir)

    async def aprocess_batch(self, sources: Iterable[Any], *, output_dir: PathLike | None = None,
                             on_result: Callable[[Result], None] | None = None) -> list[Result]:
        """:meth:`process_batch` in a worker thread (does not block the event loop)."""
        return await asyncio.to_thread(self.process_batch, list(sources), output_dir=output_dir,
                                       on_result=on_result)

    def _crash_result(self, source: Any, index: int, exc: Exception,
                      output_dir: PathLike | None) -> Result:
        from radintake.logging import get_logger
        from radintake.pipeline import _base_doc, sha256_file, write_doc

        get_logger("radintake.sdk").error("case_crashed", error_type=type(exc).__name__)
        sha = hashlib.sha256(b"").hexdigest()
        if isinstance(source, (str, os.PathLike)):
            name = _safe_input_name(Path(source).name, f"input_{index}")
            try:
                sha = sha256_file(Path(source))
            except OSError:
                pass
        else:
            name = f"input_{index}"
            if isinstance(source, (bytes, bytearray, memoryview)):
                sha = hashlib.sha256(bytes(source)).hexdigest()
        trace_id = f"{self.run_id}/crashed" if self.run_id else "crashed"
        doc = _base_doc(name, sha, "unknown", trace_id)
        if isinstance(exc, UnsupportedInputError):
            doc["error"] = f"unsupported source: {type(source).__name__}"
        else:
            doc["error"] = f"pipeline crashed: {type(exc).__name__}"
        out = Path(output_dir) if output_dir is not None else self.output_dir
        if out is not None:
            write_doc(doc, out)
        return Result.model_validate(doc)


def _read_artifacts(base: Path, rels: list[str], suffix: str | None) -> list[bytes]:
    out = []
    for rel in rels:
        if suffix and not str(rel).lower().endswith(suffix):
            continue
        p = Path(base) / rel
        if p.is_file():
            out.append(p.read_bytes())
    return out


def process(source: Any, *, config: RadintakeConfig | Config | None = None,
            input_name: str | None = None, output_dir: PathLike | None = None,
            trace_dir: PathLike | None = None, data_dir: PathLike | None = None) -> Result:
    """One-shot convenience: ``Pipeline(config, ...).process(source)``."""
    with Pipeline(config, trace_dir=trace_dir, data_dir=data_dir) as pipe:
        return pipe.process(source, input_name=input_name, output_dir=output_dir)


async def aprocess(source: Any, **kw: Any) -> Result:
    """Async :func:`process` (runs in a worker thread)."""
    return await asyncio.to_thread(process, source, **kw)


__all__ = ["BudgetConfig", "ConfigError", "ModelConfig", "Pipeline", "RadintakeConfig",
           "RadintakeError", "UnsupportedInputError", "aprocess", "process"]
