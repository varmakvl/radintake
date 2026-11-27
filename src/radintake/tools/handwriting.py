"""Local handwriting reader for PHI crops in the sign-off zone (TrOCR-small-handwritten, MIT).

Tesseract reads handwritten technologist / radiologist sign-offs poorly, so the PHI agent's
rules never see the name. The PHI agent (:mod:`radintake.agents.phi`) cuts small crops next to
credential anchors (``RT``, ``MD`` ...) and from low-confidence lines in the sign-off zone, AFTER
its masking rules ran, and hands them to :class:`HandwritingReader`; the recognised text goes back
through :class:`~radintake.agents.phi.PhiDetector` and any name found is masked on the page.

Guard decision (see ``okf/subsystems/model-client-guard.md``)
-------------------------------------------------------------
A crop read by this model is PHI *by design* - finding PHI in it is the point - so the usual
``phi_masked=True`` rule cannot apply. The reader is part of the PHI agent's trust boundary instead:

* it accepts only a :class:`PhiCrop` stamped ``phi_crop=True`` (built by the PHI agent), never a
  page, a :class:`~radintake.tools.model_client.MaskedPayload` or a raw array;
* its model must be ``kind: local`` - an ``api`` model is refused whatever ``RADINTAKE_OFFLINE``
  says (:class:`NonLocalPhiReader`), so a crop can never leave the machine;
* ``RADINTAKE_FORBID_MODEL_CALLS=1`` still refuses the call (replay mode);
* the weights are loaded with ``local_files_only=True`` and ``HF_HUB_OFFLINE=1``: never downloaded
  at run time. ``transformers`` / ``torch`` are NOT in ``requirements.lock`` (optional extra
  ``[handwriting]``); when either is missing or the weights are not cached the reader is
  unavailable and the PHI stage records a skip reason and carries on unchanged (the CI path).

Every call goes through :meth:`ModelClient.call`, so the case trace gets model id, prompt id/hash,
payload hash, output hash and timing - never the recognised text.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from radintake.config import Config
from radintake.tools.model_client import (
    ModelCallForbidden,
    ModelClient,
    ModelClientError,
    UnmaskedPayloadError,
)
from radintake.tracing.writer import CaseTrace

DEFAULT_MODEL = "hw_reader"  # registry name in configs/models.yaml
PROMPT_ID = "hw_line_v1"
PROMPT = "trocr: transcribe one handwritten text line (no text prompt; image only)"


class HandwritingUnavailable(RuntimeError):
    """The reader cannot run here; ``reason`` is a short machine-readable code."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class NonLocalPhiReader(ModelClientError):
    """A PHI-reading model must be ``kind: local``; crops never leave the machine."""


@dataclass
class PhiCrop:
    """A crop of a page that may contain PHI, cut by the PHI agent for a LOCAL reader only."""

    image: np.ndarray
    phi_crop: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def data(self) -> np.ndarray:  # hashed by ModelClient for the trace
        return self.image


def availability(model_id: str) -> str | None:
    """``None`` when the reader can run, else the skip reason.

    Cheap: import specs and the local HF cache only (no torch import, no network).
    """
    if importlib.util.find_spec("transformers") is None:
        return "transformers_missing"
    if importlib.util.find_spec("torch") is None:
        return "torch_missing"
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001 - comes with transformers; treat a broken install as missing
        return "transformers_missing"
    try:
        cfg = try_to_load_from_cache(model_id, "config.json")
        weights = [try_to_load_from_cache(model_id, f) for f in ("model.safetensors", "pytorch_model.bin")]
    except Exception:  # noqa: BLE001
        return "weights_not_cached"
    if not isinstance(cfg, str) or not any(isinstance(w, str) for w in weights):
        return "weights_not_cached"
    return None


_LOADED: dict[str, tuple[Any, Any, str]] = {}


def _load(model_id: str) -> tuple[Any, Any, str]:
    """Load (once per process) the TrOCR model + processor from the local cache only."""
    if model_id not in _LOADED:
        reason = availability(model_id)
        if reason:
            raise HandwritingUnavailable(reason)
        # Never touch the network: with only pytorch_model.bin cached, transformers otherwise starts a
        # background safetensors-conversion lookup on huggingface.co even with local_files_only=True
        # (seen 2026-11-27). The hub reads HF_HUB_OFFLINE once at import, so the flag is forced on the
        # constants module for the duration of the load (other in-process HF users are unaffected).
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        from huggingface_hub import constants as hub_constants

        was_offline = hub_constants.HF_HUB_OFFLINE
        hub_constants.HF_HUB_OFFLINE = True
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            processor = TrOCRProcessor.from_pretrained(model_id, local_files_only=True)
            model = VisionEncoderDecoderModel.from_pretrained(model_id, local_files_only=True,
                                                              dtype=dtype).to(device).eval()
        except Exception as exc:  # broken install / partial cache -> skip, never fail
            raise HandwritingUnavailable(f"load_failed:{type(exc).__name__}") from exc
        finally:
            hub_constants.HF_HUB_OFFLINE = was_offline
        _LOADED[model_id] = (model, processor, device)
    return _LOADED[model_id]


def gpu_peak_mb() -> float | None:
    """Peak CUDA memory allocated by this process (MB), or None without torch / CUDA."""
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


class HandwritingReader(ModelClient):
    """Guarded local handwriting recogniser: one :class:`PhiCrop` (one text line) -> text."""

    def __init__(self, model: str = DEFAULT_MODEL, kind: str | None = None, *,
                 config: Config | None = None, trace: CaseTrace | None = None, max_new_tokens: int = 32):
        super().__init__(model, kind, config=config, trace=trace)
        self.max_new_tokens = int(max_new_tokens)

    def unavailable_reason(self) -> str | None:
        if self.kind != "local":
            return "model_not_local"
        return availability(self.model_id)

    def check(self, payload: Any) -> None:
        if self._forbidden():
            raise ModelCallForbidden(f"model calls are forbidden (RADINTAKE_FORBID_MODEL_CALLS=1): "
                                     f"{self.model_id}")
        if self.kind != "local":
            raise NonLocalPhiReader(f"PHI crops are read by local models only; {self.model_id} is "
                                    f"kind={self.kind}")
        if not (isinstance(payload, PhiCrop) and payload.phi_crop is True):
            raise UnmaskedPayloadError(f"{self.model_id} reads only PhiCrop payloads cut by the PHI agent")

    def read(self, crop: PhiCrop, *, trace: CaseTrace | None = None) -> str:
        return str(self.call(crop, prompt_id=PROMPT_ID, prompt=PROMPT, trace=trace) or "")

    def _invoke(self, payload: Any, prompt: str, **kwargs: Any) -> str:
        model, processor, device = _load(self.model_id)
        import torch
        from PIL import Image

        # greyscale first: on coloured ink (blue pen) TrOCR-small returned unrelated text, on the same
        # crop in grey it read the name (measured 2026-11-27)
        img = Image.fromarray(np.asarray(payload.image)).convert("L").convert("RGB")
        pixels = processor(images=img, return_tensors="pt").pixel_values.to(device, dtype=model.dtype)
        with torch.inference_mode():
            ids = model.generate(pixels, max_new_tokens=self.max_new_tokens, num_beams=1, do_sample=False)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
