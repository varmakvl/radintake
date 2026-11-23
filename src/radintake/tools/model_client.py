"""Guarded base for every model call (OCR models, local VLMs, API VLMs, embedders).

Every call goes through :meth:`ModelClient.call`, which enforces, in order:

1. ``RADINTAKE_FORBID_MODEL_CALLS=1`` -> :class:`ModelCallForbidden` (replay / re-score mode),
2. ``RADINTAKE_OFFLINE=1`` and an ``api`` model -> :class:`OfflineViolation`,
3. the payload must be stamped ``phi_masked=True`` by the PHI agent -> else
   :class:`UnmaskedPayloadError`,

then times the backend call and records model id, prompt id + hash, payload/output hashes and
timings into the case trace. Subclasses implement :meth:`_invoke` only.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from radintake.config import Config, env_flag, load_config
from radintake.tracing.writer import CaseTrace


class ModelClientError(RuntimeError):
    """Base class for refused model calls."""


class ModelCallForbidden(ModelClientError):
    """RADINTAKE_FORBID_MODEL_CALLS=1: no model may be called (replay mode)."""


class OfflineViolation(ModelClientError):
    """RADINTAKE_OFFLINE=1 and the model is a network/API model."""


class UnmaskedPayloadError(ModelClientError):
    """The payload was not stamped phi_masked=True by the PHI agent."""


@dataclass
class MaskedPayload:
    """A model input that the PHI agent has masked. Only the PHI agent should build these."""

    data: Any
    phi_masked: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def is_phi_masked(payload: Any) -> bool:
    if isinstance(payload, dict):
        return payload.get("phi_masked") is True
    return getattr(payload, "phi_masked", False) is True


def prompt_hash(prompt: str) -> str:
    return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _payload_hash(payload: Any) -> str:
    data = payload.get("data", payload) if isinstance(payload, dict) else getattr(payload, "data", payload)
    h = hashlib.sha256()
    if isinstance(data, (bytes, bytearray)):
        h.update(bytes(data))
    elif hasattr(data, "tobytes"):
        h.update(data.tobytes())
    else:
        try:
            h.update(json.dumps(data, sort_keys=True, default=repr).encode("utf-8"))
        except (TypeError, ValueError):
            h.update(repr(data).encode("utf-8"))
    return "sha256:" + h.hexdigest()


class ModelClient:
    """Base guarded client. ``kind`` is ``"local"`` or ``"api"``."""

    def __init__(self, model_id: str, kind: str | None = None, *, config: Config | None = None,
                 trace: CaseTrace | None = None):
        self.config = config or load_config()
        spec = self.config.model(model_id)
        self.model_id = spec.id if spec else model_id
        self.kind = kind or (spec.kind if spec else None)
        if self.kind not in ("local", "api"):
            raise ValueError(f"unknown model kind for {model_id!r}: pass kind='local' or 'api'")
        self.trace = trace

    # The env is re-read at call time so a flag set after construction still applies.
    def _forbidden(self) -> bool:
        return env_flag("RADINTAKE_FORBID_MODEL_CALLS", self.config.forbid_model_calls)

    def _offline(self) -> bool:
        return env_flag("RADINTAKE_OFFLINE", self.config.offline)

    def check(self, payload: Any) -> None:
        if self._forbidden():
            raise ModelCallForbidden(f"model calls are forbidden (RADINTAKE_FORBID_MODEL_CALLS=1): "
                                     f"{self.model_id}")
        if self.kind == "api" and self._offline():
            raise OfflineViolation(f"API model {self.model_id} refused: RADINTAKE_OFFLINE=1")
        if not is_phi_masked(payload):
            raise UnmaskedPayloadError(f"payload for {self.model_id} is not stamped phi_masked=True")

    def call(self, payload: Any, *, prompt_id: str, prompt: str = "", trace: CaseTrace | None = None,
             **kwargs: Any) -> Any:
        self.check(payload)
        tr = trace or self.trace
        p_hash = prompt_hash(prompt)
        t0 = time.perf_counter()
        status, out = "ok", None
        try:
            out = self._invoke(payload, prompt, **kwargs)
            return out
        except Exception as exc:
            status = f"error:{type(exc).__name__}"
            raise
        finally:
            ms = round((time.perf_counter() - t0) * 1000.0, 3)
            if tr is not None:
                tr.model_call(self.model_id, prompt_id, p_hash, ms, model_kind=self.kind,
                              status=status, payload_hash=_payload_hash(payload),
                              output_hash=_payload_hash({"data": out}) if status == "ok" else None)

    def _invoke(self, payload: Any, prompt: str, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError
