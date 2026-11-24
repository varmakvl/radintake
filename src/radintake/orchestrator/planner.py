"""Planner: a small state machine that runs a case's stage plan under budgets.

States per stage: ``pending -> running -> (ok | failed | timeout | skipped)``; per case:
``running -> (ok | needs_review | error)``.

Budgets (``configs/default.yaml`` ``budgets``):

* ``max_steps`` - maximum stage attempts per case (retries count as steps);
* ``stage_timeout_s`` - wall-clock limit per attempt (a stage may override it);
* ``retries`` + ``backoff_s`` - extra attempts after an exception, sleeping
  ``backoff_s * 2**k`` between them (a stage may override ``retries``);
* ``case_deadline_s`` - global wall-clock deadline for the case.

Outcome rules (never raise to the caller):

* exception in a ``critical`` stage after retries -> case ``error`` with the message;
* exception in a non-critical stage -> ``needs_review`` + the stage's ``failure_flag``, and the
  plan continues with the next stage;
* timeout, step budget or deadline exhausted -> flag ``timeout``; ``needs_review`` if an
  earlier stage produced partial results, else ``error``. A timed-out critical stage, or an
  exhausted case budget, stops the plan.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from radintake.logging import get_logger
from radintake.orchestrator.stages import Stage, StageContext

log = get_logger("radintake.planner")


class StageTimeout(Exception):
    pass


@dataclass
class CaseOutcome:
    status: str  # ok | needs_review | error
    error: str | None = None
    steps: int = 0
    stage_status: dict[str, str] = field(default_factory=dict)


def _short(exc: BaseException) -> str:
    msg = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def run_with_timeout(fn, ctx: StageContext, timeout_s: float) -> Any:
    """Run ``fn(ctx)`` in a daemon thread; raise :class:`StageTimeout` after ``timeout_s``.

    Python threads cannot be killed: a timed-out stage is abandoned (``ctx.cancelled`` is set
    so cooperative stages can stop) and its result is discarded.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["result"] = fn(ctx)
        except BaseException as exc:  # noqa: BLE001 - forwarded to the planner thread
            box["exc"] = exc

    th = threading.Thread(target=target, name="radintake-stage", daemon=True)
    th.start()
    th.join(max(0.0, timeout_s))
    if th.is_alive():
        ctx.cancelled.set()
        raise StageTimeout(f"stage exceeded {timeout_s:.1f}s")
    if "exc" in box:
        raise box["exc"]
    return box.get("result")


def _inspect(name: str, ctx: StageContext, updates: dict[str, Any] | None, t0: float) -> None:
    """Hand a finished stage to the step inspector when RADINTAKE_INSPECT_DIR is set (debug only)."""
    import os

    if not os.environ.get("RADINTAKE_INSPECT_DIR"):
        return
    try:
        from radintake.inspect import dump_stage

        dump_stage(name, ctx, updates, round((time.perf_counter() - t0) * 1000.0, 3))
    except Exception as exc:  # noqa: BLE001 - inspection must never change the case outcome
        log.warning("inspect_failed", stage=name, error_type=type(exc).__name__)


def _merge(ctx: StageContext, updates: dict[str, Any] | None) -> None:
    if not updates:
        return
    if not isinstance(updates, dict):
        raise TypeError(f"stage must return dict or None, got {type(updates).__name__}")
    for k, v in updates.items():
        if k == "flags":
            ctx.flags.update(v or [])
        else:
            ctx.data[k] = v


class Planner:
    def __init__(self, config: Any):
        self.budgets = config.budgets

    def run(self, ctx: StageContext, plan: list[Stage]) -> CaseOutcome:
        b = self.budgets
        out = CaseOutcome(status="running")
        progressed = False  # some stage succeeded -> partial results exist
        fatal: str | None = None
        fatal_is_timeout = False
        if ctx.trace is not None:
            ctx.trace.event("plan", input_type=ctx.input_type, plan=[s.name for s in plan])

        for stage in plan:
            if fatal:
                out.stage_status[stage.name] = "skipped"
                continue
            if ctx.remaining_s() <= 0 or out.steps >= b.max_steps:
                ctx.flags.add("timeout")
                reason = "case deadline exceeded" if ctx.remaining_s() <= 0 else "step budget exhausted"
                fatal, fatal_is_timeout = reason, True
                out.stage_status[stage.name] = "skipped"
                continue

            retries = b.retries if stage.retries is None else stage.retries
            timeout = b.stage_timeout_s if stage.timeout_s is None else stage.timeout_s
            attempt, status, err = 0, "failed", None
            while True:
                attempt += 1
                out.steps += 1
                budget = min(timeout, max(0.0, ctx.remaining_s()))
                t0 = time.perf_counter()
                try:
                    updates = run_with_timeout(stage.fn, ctx, budget)
                    _merge(ctx, updates)
                    status, err = "ok", None
                    _inspect(stage.name, ctx, updates, t0)
                except StageTimeout as exc:
                    status, err = "timeout", _short(exc)
                except Exception as exc:  # noqa: BLE001 - recorded, never propagated
                    status, err = "failed", _short(exc)
                ms = round((time.perf_counter() - t0) * 1000.0, 3)
                if ctx.trace is not None:
                    ctx.trace.step(stage.name, status, ms, attempt=attempt,
                                   error_type=err.split(":")[0] if err else None)
                log.info("stage_done", stage=stage.name, status=status, attempt=attempt, duration_ms=ms)
                if status != "failed" or attempt > retries or out.steps >= b.max_steps:
                    break
                sleep = min(b.backoff_s * (2 ** (attempt - 1)), max(0.0, ctx.remaining_s()))
                time.sleep(sleep)
            out.stage_status[stage.name] = status

            if status == "ok":
                progressed = True
            elif status == "timeout":
                ctx.flags.add("timeout")
                if stage.critical:
                    fatal, fatal_is_timeout = f"stage {stage.name} timed out: {err}", True
                else:
                    ctx.data.setdefault("review_reasons", []).append(f"{stage.name}: timeout")
            elif stage.critical:
                fatal = f"stage {stage.name} failed: {err}"
            else:
                if stage.failure_flag:
                    ctx.flags.add(stage.failure_flag)
                ctx.data.setdefault("review_reasons", []).append(f"{stage.name}: {err}")

        if fatal:
            # partial results (an earlier stage succeeded) + a timeout/budget stop -> review, not error
            out.status = "needs_review" if (fatal_is_timeout and progressed) else "error"
            out.error = fatal
        else:
            out.status = "needs_review" if (ctx.flags or ctx.data.get("review_reasons")) else "ok"
        return out
