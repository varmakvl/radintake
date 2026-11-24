"""Local-only PHI vault: typed placeholders <-> original values.

The PHI agent replaces every detected PHI span with a typed placeholder such as
``[PATIENT_NAME_1]`` and records the mapping here, so header fields can still be written to the
LOCAL structured output after extraction ran on masked text. The vault never leaves the
process: it is not serialised into outputs or traces, and every value put in it is registered
with the log/trace scrubber (:func:`radintake.logging.register_phi`), so a value that slipped
into a log line or trace payload is redacted.
"""

from __future__ import annotations

import re
import threading

from radintake.logging import register_phi

_PLACEHOLDER = re.compile(r"\[([A-Z][A-Z_]*)_(\d+)\]")
_MIN_TOKEN = 3


def _norm(value: str) -> str:
    return " ".join(value.split()).casefold()


class PhiVault:
    """Maps placeholders to PHI values. The same (type, value) always gets the same placeholder."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_key: dict[tuple[str, str], str] = {}
        self._values: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def placeholder(self, phi_type: str, value: str) -> str:
        """Return the placeholder for ``value`` (allocating one) and register it as PHI."""
        t = re.sub(r"[^A-Z_]", "_", phi_type.upper()) or "PHI"
        key = (t, _norm(value))
        with self._lock:
            ph = self._by_key.get(key)
            if ph is None:
                n = self._counters.get(t, 0) + 1
                self._counters[t] = n
                ph = f"[{t}_{n}]"
                self._by_key[key] = ph
                self._values[ph] = value
        register_phi(value)
        # name parts are PHI on their own ("Quistgaard" alone in a log line is still a leak)
        if t in ("PATIENT_NAME", "PERSON"):
            register_phi(*[p for p in re.split(r"[\s,]+", value) if len(p) >= _MIN_TOKEN])
        return ph

    def value(self, placeholder: str) -> str | None:
        with self._lock:
            return self._values.get(placeholder)

    def restore(self, text: str) -> str:
        """Replace every known placeholder in ``text`` by its original value (local use only)."""
        if not text:
            return text
        with self._lock:
            values = dict(self._values)
        return _PLACEHOLDER.sub(lambda m: values.get(m.group(0), m.group(0)), text)

    def placeholders(self) -> dict[str, str]:
        """Placeholder -> type (no values: safe to log counts from)."""
        with self._lock:
            return {ph: ph[1:].rsplit("_", 1)[0] for ph in self._values}

    def values_of(self, phi_type: str) -> list[str]:
        t = phi_type.upper()
        with self._lock:
            return [v for (tt, _), ph in self._by_key.items() if tt == t for v in [self._values[ph]]]

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    def __repr__(self) -> str:  # never print values
        return f"PhiVault(n={len(self)})"
