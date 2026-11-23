"""Guarded VLM client: PHI-masked page images / crops in, JSON out; cached, traced, costed.

:class:`VlmClient` subclasses the guarded :class:`~radintake.tools.model_client.ModelClient`, so
every live call is refused unless the payload is stamped ``phi_masked=True`` by the PHI agent,
refused under ``RADINTAKE_FORBID_MODEL_CALLS=1``, and refused for an ``api`` model under
``RADINTAKE_OFFLINE=1``. On top of that it adds:

* **response cache** keyed by ``sha256(model id + payload hash + prompt hash)`` (JSON files under
  ``extraction.vlm.cache_dir``, default ``<data_dir>/cache/vlm``) so runs can be replayed for free;
  a cache hit is not a model call and still requires a masked payload;
* **trace** per call: model id + kind, prompt id + hash, payload/output hashes, latency, token
  usage, cost (USD, from ``configs/models.yaml`` pricing) and ``cache_hit`` - never content;
* **backends**: ``api`` = Anthropic Messages API over HTTPS (key from the ``ANTHROPIC_API_KEY``
  environment variable only, registered with the log scrubber, never traced); ``local`` = a small
  local VLM via ``transformers`` (Florence-2-base, crop transcription only) when it is installed
  and its weights are already cached - otherwise :func:`local_available` is False.

Use :func:`select_vlm` to get the client the run may use (or ``None`` plus the reason).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from radintake.config import REPO_ROOT, Config, env_flag, load_config
from radintake.logging import register_phi
from radintake.tools.model_client import (
    MaskedPayload,
    ModelClient,
    UnmaskedPayloadError,
    _payload_hash,
    is_phi_masked,
    prompt_hash,
)
from radintake.tracing.writer import CaseTrace

API_KEY_ENV = "ANTHROPIC_API_KEY"
API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
PROMPTS_DIR = REPO_ROOT / "prompts"


class VlmUnavailable(RuntimeError):
    """No usable backend (no API key, or local model not installed / not cached)."""


@dataclass
class VlmResponse:
    data: Any  # parsed JSON object from the model
    usage: dict[str, int] = field(default_factory=dict)  # input_tokens / output_tokens
    cost_usd: float = 0.0
    cached: bool = False


def load_prompt(prompt_id: str) -> str:
    """Prompt text of ``prompts/<prompt_id>.md``."""
    return (PROMPTS_DIR / f"{prompt_id}.md").read_text(encoding="utf-8")


def vlm_params(cfg: Config) -> dict[str, Any]:
    raw = cfg.raw or {}
    ext = raw.get("extraction") if isinstance(raw.get("extraction"), dict) else {}
    v = ext.get("vlm") if isinstance(ext.get("vlm"), dict) else {}
    return {"model": "vlm_api", "local_model": "vlm_local", "max_tokens": 2000, "timeout_s": 60.0,
            "cache_dir": None, **v}


def _model_spec_raw(cfg: Config, model_id: str) -> dict[str, Any]:
    for name, spec in (cfg.raw.get("models") or {}).items():
        if isinstance(spec, dict) and (name == model_id or spec.get("id") == model_id):
            return spec
    return {}


def parse_json(text: str) -> Any:
    """First JSON object in a model reply (tolerates code fences / prose around it)."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model reply holds no JSON object")
    return json.loads(text[start:end + 1])


def _png_b64(img: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(img)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def crop_payload(payload: Any, box: Any, *, pad: int = 8) -> MaskedPayload:
    """A crop of an already PHI-masked page payload; refuses an unmasked source."""
    if not is_phi_masked(payload):
        raise UnmaskedPayloadError("crop source is not stamped phi_masked=True")
    data = payload.data if isinstance(payload, MaskedPayload) else payload.get("data")
    img = np.asarray(data["image"])
    h, w = img.shape[:2]
    x0, y0 = max(0, int(box.x0) - pad), max(0, int(box.y0) - pad)
    x1, y1 = min(w, int(np.ceil(box.x1)) + pad), min(h, int(np.ceil(box.y1)) + pad)
    meta = dict(getattr(payload, "meta", {}) or {})
    meta.update(derived="crop", crop=[x0, y0, x1, y1])
    return MaskedPayload(data={"image": np.ascontiguousarray(img[y0:y1, x0:x1]), "page": data.get("page"),
                               "offset": [x0, y0]}, phi_masked=True, meta=meta)


class VlmClient(ModelClient):
    """Guarded VLM client with response cache and usage/cost tracing."""

    def __init__(self, model_id: str, kind: str | None = None, *, config: Config | None = None,
                 trace: CaseTrace | None = None, cache_dir: str | Path | None = None):
        super().__init__(model_id, kind, config=config, trace=trace)
        p = vlm_params(self.config)
        cdir = cache_dir or p.get("cache_dir") or (Path(self.config.data_dir) / "cache" / "vlm")
        self.cache_dir = Path(cdir)
        self.max_tokens = int(p["max_tokens"])
        self.timeout_s = float(p["timeout_s"])
        spec = _model_spec_raw(self.config, self.model_id)
        self.price_in = float(spec.get("usd_per_mtok_in", 0.0) or 0.0)
        self.price_out = float(spec.get("usd_per_mtok_out", 0.0) or 0.0)

    # -- cache --------------------------------------------------------------------------
    def cache_key(self, payload: Any, prompt: str) -> str:
        raw = f"{self.model_id}|{_payload_hash(payload)}|{prompt_hash(prompt)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> VlmResponse | None:
        p = self.cache_dir / f"{key}.json"
        if not p.is_file():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return VlmResponse(d["data"], d.get("usage") or {}, float(d.get("cost_usd") or 0.0), cached=True)
        except (OSError, ValueError, KeyError):
            return None

    def _cache_put(self, key: str, resp: VlmResponse) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_dir / f"{key}.json.tmp"
            tmp.write_text(json.dumps({"data": resp.data, "usage": resp.usage, "cost_usd": resp.cost_usd}),
                           encoding="utf-8")
            os.replace(tmp, self.cache_dir / f"{key}.json")
        except OSError:
            pass  # a cache that cannot be written never fails the case

    # -- call ---------------------------------------------------------------------------
    def call(self, payload: Any, *, prompt_id: str, prompt: str = "", trace: CaseTrace | None = None,
             **kwargs: Any) -> VlmResponse:
        if not is_phi_masked(payload):  # before cache AND backend: nothing unmasked is ever used
            raise UnmaskedPayloadError(f"payload for {self.model_id} is not stamped phi_masked=True")
        tr = trace or self.trace
        p_hash = prompt_hash(prompt)
        key = self.cache_key(payload, prompt)
        t0 = time.perf_counter()
        resp = self._cache_get(key)
        status = "ok"
        if resp is None:
            self.check(payload)  # forbid / offline / mask guards before any live call
            try:
                resp = self._invoke(payload, prompt, **kwargs)
                self._cache_put(key, resp)
            except Exception as exc:
                status = f"error:{type(exc).__name__}"
                raise
            finally:
                if status != "ok":
                    self._trace(tr, prompt_id, p_hash, t0, payload, None, status)
        self._trace(tr, prompt_id, p_hash, t0, payload, resp, status)
        return resp

    def _trace(self, tr, prompt_id, p_hash, t0, payload, resp: VlmResponse | None, status: str) -> None:
        if tr is None:
            return
        usage = (resp.usage if resp else {}) or {}
        tr.model_call(self.model_id, prompt_id, p_hash, round((time.perf_counter() - t0) * 1000.0, 3),
                      model_kind=self.kind, status=status, payload_hash=_payload_hash(payload),
                      output_hash=_payload_hash({"data": resp.data}) if resp else None,
                      input_tokens=int(usage.get("input_tokens", 0)),
                      output_tokens=int(usage.get("output_tokens", 0)),
                      cost_usd=round(resp.cost_usd, 6) if resp else 0.0,
                      cache_hit=bool(resp and resp.cached))

    def cost(self, usage: dict[str, int]) -> float:
        return (usage.get("input_tokens", 0) * self.price_in + usage.get("output_tokens", 0) * self.price_out) / 1e6

    def _invoke(self, payload: Any, prompt: str, **kwargs: Any) -> VlmResponse:
        if self.kind == "api":
            return self._invoke_api(payload, prompt, **kwargs)
        return self._invoke_local(payload, prompt, **kwargs)

    # -- backends -----------------------------------------------------------------------
    def _invoke_api(self, payload: Any, prompt: str, **kwargs: Any) -> VlmResponse:
        import urllib.request

        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise VlmUnavailable(f"{API_KEY_ENV} is not set")
        register_phi(key)  # defence in depth: a key that slipped into a log line is redacted
        data = payload.data if isinstance(payload, MaskedPayload) else payload["data"]
        content: list[dict[str, Any]] = [{"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": _png_b64(data["image"])}}]
        content.append({"type": "text", "text": prompt})
        body = json.dumps({"model": self.model_id, "max_tokens": self.max_tokens, "temperature": 0,
                           "messages": [{"role": "user", "content": content}]}).encode("utf-8")
        req = urllib.request.Request(API_URL, data=body, method="POST", headers={
            "x-api-key": key, "anthropic-version": API_VERSION, "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            out = json.loads(r.read().decode("utf-8"))
        text = "".join(c.get("text", "") for c in out.get("content", []) if c.get("type") == "text")
        u = out.get("usage") or {}
        usage = {"input_tokens": int(u.get("input_tokens", 0)), "output_tokens": int(u.get("output_tokens", 0))}
        return VlmResponse(parse_json(text), usage, self.cost(usage))

    def _invoke_local(self, payload: Any, prompt: str, **kwargs: Any) -> VlmResponse:
        """Florence-2 ``<OCR>`` on a crop -> ``{"text": ...}``. Page-level field JSON is API-only."""
        if kwargs.get("task") != "crop":
            raise VlmUnavailable("local VLM supports crop transcription only")
        model, processor, device = _local_model(self.model_id)
        from PIL import Image

        data = payload.data if isinstance(payload, MaskedPayload) else payload["data"]
        img = Image.fromarray(np.asarray(data["image"])).convert("RGB")
        inputs = processor(text="<OCR>", images=img, return_tensors="pt").to(device)
        ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                             max_new_tokens=256, num_beams=3, do_sample=False)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        usage = {"input_tokens": int(inputs["input_ids"].shape[-1]), "output_tokens": int(ids.shape[-1])}
        return VlmResponse({"text": text, "confidence": 0.7}, usage, 0.0)


_LOCAL: dict[str, Any] = {}


def local_available(model_id: str) -> bool:
    """True when transformers is installed and the weights are already in the local HF cache."""
    try:
        import transformers  # noqa: F401
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001 - not installed -> unavailable
        return False
    try:
        return isinstance(try_to_load_from_cache(model_id, "config.json"), str)
    except Exception:  # noqa: BLE001
        return False


def _local_model(model_id: str):
    if model_id not in _LOCAL:
        if not local_available(model_id):
            raise VlmUnavailable(f"local VLM {model_id} is not installed / cached")
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download at run time
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype,
                                                     local_files_only=True).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        _LOCAL[model_id] = (model, processor, device)
    return _LOCAL[model_id]


def select_vlm(cfg: Config | None = None, trace: CaseTrace | None = None, *,
               task: str = "page") -> tuple[VlmClient | None, str]:
    """The VLM client this run may use for ``task`` (``page`` | ``crop``), or ``(None, reason)``.

    Offline (``RADINTAKE_OFFLINE=1``) the API model is never chosen; the local VLM is used if it is
    installed and cached (crop task only), otherwise ``None`` - callers fall back to OCR-first.
    """
    cfg = cfg or load_config()
    if env_flag("RADINTAKE_FORBID_MODEL_CALLS", cfg.forbid_model_calls):
        return None, "model_calls_forbidden"
    p = vlm_params(cfg)
    offline = env_flag("RADINTAKE_OFFLINE", cfg.offline)
    api = cfg.model(p["model"])
    local = cfg.model(p["local_model"])
    if not offline and api is not None and os.environ.get(API_KEY_ENV):
        return VlmClient(api.id, api.kind, config=cfg, trace=trace), "api"
    if local is not None and task == "crop" and local_available(local.id):
        return VlmClient(local.id, "local", config=cfg, trace=trace), "local"
    if offline:
        return None, "offline_api_unavailable_no_local_vlm"
    return None, "no_api_key_no_local_vlm"
