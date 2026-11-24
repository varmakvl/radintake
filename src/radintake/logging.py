"""Structured JSON logging to stderr with a PHI scrubber.

Every log line passes through :func:`scrub_processor`, which replaces any value registered
with :func:`register_phi` (and anything matched by a registered scrubber hook, e.g. the PHI
vault later) by ``[REDACTED]``. The same registry is used by the trace writer, so there is
one place to tell the system "this string is PHI, never emit it".

Policy: log ids, hashes, counts and timings only. The scrubber is a safety net, not a
licence to log report text.
"""

from __future__ import annotations

import logging as _stdlib_logging
import re
import sys
import threading
from collections.abc import Callable, Iterable
from typing import Any

import structlog

REDACTED = "[REDACTED]"
_MIN_LEN = 3  # never register trivially short strings (would shred unrelated text)

_lock = threading.RLock()
_phi_values: set[str] = set()
_pattern: re.Pattern[str] | None = None
_hooks: list[Callable[[str], str]] = []


def register_phi(*values: str | None) -> None:
    """Register PHI strings to be scrubbed from logs and traces (case-insensitive)."""
    global _pattern
    with _lock:
        changed = False
        for v in values:
            if v is None:
                continue
            s = str(v).strip()
            if len(s) >= _MIN_LEN and s not in _phi_values:
                _phi_values.add(s)
                changed = True
        if changed:
            alts = sorted(_phi_values, key=len, reverse=True)
            _pattern = re.compile("|".join(re.escape(a) for a in alts), re.IGNORECASE)


def register_scrubber(hook: Callable[[str], str]) -> None:
    """Register a text -> text redaction hook (e.g. a vault-aware redactor)."""
    with _lock:
        _hooks.append(hook)


def clear_phi() -> None:
    """Forget all registered PHI values and hooks (tests, end of a case)."""
    global _pattern
    with _lock:
        _phi_values.clear()
        _hooks.clear()
        _pattern = None


def registered_phi() -> frozenset[str]:
    with _lock:
        return frozenset(_phi_values)


def scrub(text: str) -> str:
    """Return ``text`` with every registered PHI value (and hook match) redacted."""
    if not isinstance(text, str) or not text:
        return text
    with _lock:
        pat, hooks = _pattern, list(_hooks)
    if pat is not None:
        text = pat.sub(REDACTED, text)
    for h in hooks:
        try:
            text = h(text)
        except Exception:  # noqa: BLE001 - a broken hook must never leak: fail closed
            return REDACTED
    return text


def contains_phi(text: str) -> bool:
    return isinstance(text, str) and scrub(text) != text


def scrub_obj(obj: Any) -> Any:
    """Recursively scrub strings inside dicts/lists/tuples (keys included)."""
    if isinstance(obj, str):
        return scrub(obj)
    if isinstance(obj, dict):
        return {scrub(k) if isinstance(k, str) else k: scrub_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub_obj(v) for v in obj]
    return obj


def scrub_processor(_logger: Any, _method: str, event_dict: dict) -> dict:
    """structlog processor: scrub PHI from every value of the event dict."""
    return scrub_obj(event_dict)


class _StderrLogger:
    """Writes each rendered line to the *current* ``sys.stderr`` (capture-friendly)."""

    def msg(self, message: str) -> None:
        stream = sys.stderr
        stream.write(message + "\n")
        stream.flush()

    log = debug = info = warning = warn = error = critical = exception = fatal = msg


class _StderrLoggerFactory:
    def __call__(self, *args: Any) -> _StderrLogger:
        return _StderrLogger()


_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog: JSON lines on stderr, PHI scrubbed, level-filtered."""
    global _configured
    lvl = getattr(_stdlib_logging, str(level).upper(), _stdlib_logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            scrub_processor,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(lvl),
        logger_factory=_StderrLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    _configured = True


def get_logger(name: str = "radintake", **initial: Any):
    if not _configured:
        configure_logging()
    return structlog.get_logger(name).bind(logger=name, **initial)


def register_phi_many(values: Iterable[str | None]) -> None:
    register_phi(*list(values))
