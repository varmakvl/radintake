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
* **backends** (all local and open source, chosen per model by ``backend`` in ``configs/models.yaml``):
  ``ollama`` = a local Ollama server (``OLLAMA_HOST``, default ``http://127.0.0.1:11434``) serving
  Qwen2.5-VL-3B for page reading and crop re-reads; ``transformers`` = Florence-2-base in process,
  crop transcription only, when it is installed and its weights are already cached. No paid API is
  used; a model of kind ``api`` would still be refused offline by the base guard.

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
from radintake.tools.model_client import (
    MaskedPayload,
    ModelClient,
    UnmaskedPayloadError,
    _payload_hash,
    is_phi_masked,
    prompt_hash,
)
from radintake.tracing.writer import CaseTrace

OLLAMA_HOST_ENV = "OLLAMA_HOST"
OLLAMA_DEFAULT = "http://127.0.0.1:11434"
PROMPTS_DIR = REPO_ROOT / "prompts"


class VlmUnavailable(RuntimeError):
    """No usable backend (Ollama not running / model not pulled, local model not cached)."""


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
    return {"model": "vlm_page", "local_model": "vlm_local", "max_tokens": 2000, "timeout_s": 180.0,
            "cache_dir": None, "max_image_side": 1100, "num_ctx": 8192, **v}


def downscale(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Shrink ``img`` so its longer side is <= ``max_side`` (never enlarges). Returns (image, scale)."""
    h, w = img.shape[:2]
    if not max_side or max(h, w) <= max_side:
        return img, 1.0
    from PIL import Image

    scale = max_side / float(max(h, w))
    out = Image.fromarray(np.asarray(img)).resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                                   Image.Resampling.LANCZOS)
    return np.asarray(out), scale


def rescale_boxes(obj: Any, factor: float) -> Any:
    """Multiply every ``"bbox": [x0, y0, x1, y1]`` in a model reply by ``factor`` (back to page pixels)."""
    if factor == 1.0:
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "bbox" and isinstance(v, list) and len(v) == 4 and all(isinstance(n, (int, float)) for n in v):
                out[k] = [float(n) * factor for n in v]
            else:
                out[k] = rescale_boxes(v, factor)
        return out
    if isinstance(obj, list):
        return [rescale_boxes(v, factor) for v in obj]
    return obj


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
        self.backend = str(spec.get("backend") or ("ollama" if ":" in self.model_id else "transformers"))
        self.max_image_side = int(p.get("max_image_side") or 0)
        self.num_ctx = int(p.get("num_ctx") or 8192)

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
        if self.backend == "ollama":
            return self._invoke_ollama(payload, prompt, **kwargs)
        return self._invoke_local(payload, prompt, **kwargs)

    # -- backends -----------------------------------------------------------------------
    def _invoke_ollama(self, payload: Any, prompt: str, **kwargs: Any) -> VlmResponse:
        """Local Ollama ``/api/chat``: masked PNG + prompt in, JSON reply out (temperature 0)."""
        import urllib.error
        import urllib.request

        data = payload.data if isinstance(payload, MaskedPayload) else payload["data"]
        # A full 1700x2200 page is ~4k image tokens: it fills the context, truncates the reply and takes
        # ~60 s on 4 GB. Downscaled to max_image_side (~1.2k tokens) it reads the same text in ~5 s.
        img, scale = downscale(np.asarray(data["image"]), self.max_image_side)
        body = json.dumps({
            "model": self.model_id, "stream": False, "format": "json",
            "options": {"temperature": 0, "seed": int(self.config.seed), "num_predict": self.max_tokens,
                        "num_ctx": self.num_ctx},
            "messages": [{"role": "user", "content": prompt, "images": [_png_b64(img)]}],
        }).encode("utf-8")
        # A local server shared with other callers (or reloading the model on a 4 GB GPU) occasionally
        # returns an empty or cut-off reply; one retry turns those transient failures into answers.
        last: Exception | None = None
        for _attempt in range(2):
            req = urllib.request.Request(f"{ollama_host()}/api/chat", data=body, method="POST",
                                         headers={"content-type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    out = json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:200].decode("utf-8", "replace")
                last = VlmUnavailable(f"Ollama HTTP {exc.code}: {detail}")
                continue
            except urllib.error.URLError as exc:
                raise VlmUnavailable(f"Ollama not reachable at {ollama_host()}: {exc.reason}") from exc
            text = (out.get("message") or {}).get("content", "")
            usage = {"input_tokens": int(out.get("prompt_eval_count", 0) or 0),
                     "output_tokens": int(out.get("eval_count", 0) or 0)}
            try:
                return VlmResponse(rescale_boxes(parse_json(text), 1.0 / scale), usage, self.cost(usage))
            except ValueError as exc:  # empty / truncated / non-JSON reply
                last = exc
        raise last if last else VlmUnavailable("no reply")

    def _invoke_local(self, payload: Any, prompt: str, **kwargs: Any) -> VlmResponse:
        """Florence-2 ``<OCR>`` on a crop -> ``{"text": ...}``. Page-level field JSON needs the Ollama VLM."""
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


def ollama_host() -> str:
    host = os.environ.get(OLLAMA_HOST_ENV) or OLLAMA_DEFAULT
    if not host.startswith("http"):
        host = "http://" + host
    return host.rstrip("/")


def ollama_available(model_id: str, timeout_s: float = 2.0) -> bool:
    """True when a local Ollama server answers and ``model_id`` is pulled (no download is ever started)."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{ollama_host()}/api/tags", timeout=timeout_s) as r:
            tags = json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - server down / unreachable -> unavailable
        return False
    names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
    return model_id in names or f"{model_id}:latest" in names


def backend_available(spec: Any, raw: dict[str, Any]) -> bool:
    backend = raw.get("backend") or ("ollama" if ":" in spec.id else "transformers")
    if spec.kind != "local":
        return False  # no paid / remote API is used by this project
    return ollama_available(spec.id) if backend == "ollama" else local_available(spec.id)


def select_vlm(cfg: Config | None = None, trace: CaseTrace | None = None, *,
               task: str = "page") -> tuple[VlmClient | None, str]:
    """The VLM client this run may use for ``task`` (``page`` | ``crop``), or ``(None, reason)``.

    Order: the page VLM (Ollama, Qwen2.5-VL) for any task; for ``crop`` only, the in-process
    Florence-2 fallback. Every model is local, so offline mode does not change the choice. ``None``
    means callers fall back to OCR-first (CI and machines without Ollama).
    """
    cfg = cfg or load_config()
    if env_flag("RADINTAKE_FORBID_MODEL_CALLS", cfg.forbid_model_calls):
        return None, "model_calls_forbidden"
    p = vlm_params(cfg)
    page = cfg.model(p["model"])
    local = cfg.model(p["local_model"])
    if page is not None and backend_available(page, _model_spec_raw(cfg, page.id)):
        return VlmClient(page.id, page.kind, config=cfg, trace=trace), "ollama"
    if local is not None and task == "crop" and backend_available(local, _model_spec_raw(cfg, local.id)):
        return VlmClient(local.id, "local", config=cfg, trace=trace), "local"
    return None, "no_local_vlm_available"
