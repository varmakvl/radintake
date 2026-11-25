"""Stage registry: the per-input plan is assembled from registered stages.

Each lane registers its own stages from its own module, so parallel lanes never edit one
shared list::

    from radintake.orchestrator.stages import register_stage

    @register_stage("phi", applies_to=DOCUMENT_TYPES, order=30, failure_flag="phi_uncertain")
    def phi_stage(ctx):
        ...
        return {"phi": {...}, "flags": ["phi_uncertain"]}

Stage contract
--------------
* ``fn(ctx: StageContext) -> dict | None``. ``ctx`` is read-only for the stage; the returned
  dict is merged into ``ctx.data`` by the planner ONLY when the stage succeeds in time (a
  timed-out stage keeps running in an abandoned thread but can never corrupt the case).
  The ``flags`` key is unioned into the case flags; every other key overwrites.
* Well-known ``ctx.data`` keys: ``pages`` (list[Page]), ``n_pages``, ``report``, ``phi``,
  ``dicom``, ``review_reasons`` (list[str]).
* Stages must not print, and must log ids/hashes only.

Discovery: :func:`discover` imports every submodule of the packages in
:data:`PLUGIN_PACKAGES` (and any module named in config ``stages.plugins``); importing a
module registers its stages. A plugin that fails to import is logged and skipped.
"""

from __future__ import annotations

import importlib
import pkgutil
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DOCUMENT_TYPES = frozenset({"scan", "photo", "pdf", "tiff"})
DICOM_TYPES = frozenset({"dicom"})
ALL_TYPES = DOCUMENT_TYPES | DICOM_TYPES

PLUGIN_PACKAGES = ("radintake.agents", "radintake.stages_ext")

StageFn = Callable[["StageContext"], "dict[str, Any] | None"]


@dataclass
class StageContext:
    path: Path
    input_name: str
    input_type: str
    input_sha256: str
    out_dir: Path
    config: Any
    trace: Any  # CaseTrace
    deadline: float  # time.monotonic() deadline for the whole case
    data: dict[str, Any] = field(default_factory=dict)
    flags: set[str] = field(default_factory=set)
    cancelled: threading.Event = field(default_factory=threading.Event)

    def remaining_s(self) -> float:
        return self.deadline - time.monotonic()


@dataclass(frozen=True)
class Stage:
    name: str
    fn: StageFn
    applies_to: frozenset[str]
    order: int = 100
    timeout_s: float | None = None  # None -> config budgets.stage_timeout_s
    retries: int | None = None  # None -> config budgets.retries
    critical: bool = False  # failure -> case status "error" (else needs_review + failure_flag)
    failure_flag: str | None = None  # cli_output flag to raise when a non-critical stage fails


_REGISTRY: dict[str, Stage] = {}
_lock = threading.RLock()
_discovered = False


def register(stage: Stage, *, replace: bool = True) -> Stage:
    with _lock:
        if not replace and stage.name in _REGISTRY:
            raise ValueError(f"stage {stage.name!r} already registered")
        _REGISTRY[stage.name] = stage
    return stage


def register_stage(name: str, *, applies_to: Iterable[str], order: int = 100,
                   timeout_s: float | None = None, retries: int | None = None, critical: bool = False,
                   failure_flag: str | None = None) -> Callable[[StageFn], StageFn]:
    """Decorator form of :func:`register`."""
    types = frozenset(applies_to)
    unknown = types - ALL_TYPES
    if unknown:
        raise ValueError(f"unknown input types {sorted(unknown)}")

    def deco(fn: StageFn) -> StageFn:
        register(Stage(name=name, fn=fn, applies_to=types, order=order, timeout_s=timeout_s,
                       retries=retries, critical=critical, failure_flag=failure_flag))
        return fn

    return deco


def unregister(name: str) -> None:
    with _lock:
        _REGISTRY.pop(name, None)


def registered() -> dict[str, Stage]:
    with _lock:
        return dict(_REGISTRY)


def plan_for(input_type: str) -> list[Stage]:
    """Ordered stages that apply to ``input_type`` (ties broken by name, deterministic)."""
    with _lock:
        stages = [s for s in _REGISTRY.values() if input_type in s.applies_to]
    return sorted(stages, key=lambda s: (s.order, s.name))


def discover(extra_modules: Iterable[str] = ()) -> list[str]:
    """Import stage plugin modules once. Returns the module names imported."""
    global _discovered
    from radintake.logging import get_logger

    log = get_logger("radintake.stages")
    imported: list[str] = []
    with _lock:
        if _discovered and not extra_modules:
            return imported
        names: list[str] = []
        for pkg_name in PLUGIN_PACKAGES:
            try:
                pkg = importlib.import_module(pkg_name)
            except ModuleNotFoundError:
                continue
            names.append(pkg_name)
            names += [m.name for m in pkgutil.walk_packages(getattr(pkg, "__path__", []), pkg_name + ".")]
        names += list(extra_modules)
        for mod in names:
            try:
                importlib.import_module(mod)
                imported.append(mod)
            except Exception as exc:  # noqa: BLE001 - a broken plugin must not kill the batch
                log.warning("stage_plugin_import_failed", module=mod, error_type=type(exc).__name__)
        _discovered = True
    return imported


# --- built-in stage: page loading for document inputs -----------------------------------
@register_stage("load", applies_to=DOCUMENT_TYPES, order=10, critical=True, retries=0)
def load_stage(ctx: StageContext) -> dict[str, Any]:
    from radintake.tools.loader import load_pages

    loader_cfg = ((getattr(ctx.config, "raw", None) or {}).get("loader") or {})
    pages = load_pages(ctx.path, ctx.input_type, dpi=ctx.config.pdf_dpi,
                       text_layer=loader_cfg.get("pdf_text_layer"))
    for pg in pages:
        if "text_layer" in pg.meta:
            ctx.trace.event("pdf_text_layer", page=pg.index, **pg.meta["text_layer"])
    # text a reader cannot see (white / tiny / off-page) is dropped: it is a classic prompt-injection
    # channel, so the case goes to review under the existing v1 flag, with a specific reason code
    hidden = sum((pg.meta.get("text_layer") or {}).get("hidden_words", 0) for pg in pages)
    out: dict[str, Any] = {"pages": pages, "n_pages": len(pages)}
    if hidden:
        out["flags"] = ["prompt_injection_suspected"]
        out["review_reasons"] = [*(ctx.data.get("review_reasons") or []),
                                 f"hidden_text: {hidden} invisible text-layer words dropped"]
    return out
