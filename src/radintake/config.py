"""Runtime configuration: YAML files plus environment overrides.

Precedence (lowest -> highest):

1. ``DEFAULTS`` below (so the package works even without the ``configs/`` folder),
2. ``configs/default.yaml`` and ``configs/models.yaml`` (repo root, or ``RADINTAKE_CONFIG_DIR``),
3. an optional user YAML (``--config``),
4. environment variables ``RADINTAKE_OFFLINE``, ``RADINTAKE_DATA_DIR``, ``RADINTAKE_TRACE_DIR``,
   ``RADINTAKE_FORBID_MODEL_CALLS``.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS: dict[str, Any] = {
    "seed": 140,
    "paths": {"data_dir": "data", "trace_dir": None},
    "runtime": {"offline": False, "forbid_model_calls": False},
    "budgets": {
        "max_steps": 20,
        "stage_timeout_s": 300.0,
        "retries": 1,
        "backoff_s": 0.5,
        "case_deadline_s": 900.0,
    },
    "loader": {"pdf_dpi": 200},
    "logging": {"level": "INFO"},
    "models": {},
}

_TRUE = {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in _TRUE


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"config file {path.name} must contain a mapping")
    return data


def config_dir() -> Path:
    env = os.environ.get("RADINTAKE_CONFIG_DIR")
    return Path(env) if env else REPO_ROOT / "configs"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    id: str
    kind: str  # "local" | "api"


@dataclass(frozen=True)
class Budgets:
    max_steps: int
    stage_timeout_s: float
    retries: int
    backoff_s: float
    case_deadline_s: float


@dataclass
class Config:
    offline: bool
    forbid_model_calls: bool
    data_dir: Path
    trace_dir: Path | None
    budgets: Budgets
    pdf_dpi: int
    seed: int
    log_level: str
    models: dict[str, ModelSpec] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def model(self, name_or_id: str) -> ModelSpec | None:
        """Look a model up by registry name or by model id."""
        if name_or_id in self.models:
            return self.models[name_or_id]
        for spec in self.models.values():
            if spec.id == name_or_id:
                return spec
        return None

    @property
    def model_ids(self) -> dict[str, str]:
        return {k: v.id for k, v in self.models.items()}


def load_config(path: str | Path | None = None) -> Config:
    """Build the effective config. ``path`` is an optional user YAML (``--config``)."""
    cdir = config_dir()
    raw = _deep_merge(DEFAULTS, _read_yaml(cdir / "default.yaml"))
    raw = _deep_merge(raw, _read_yaml(cdir / "models.yaml"))
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"config file not found: {p}")
        raw = _deep_merge(raw, _read_yaml(p))

    paths, runtime, b = raw["paths"], raw["runtime"], raw["budgets"]
    offline = env_flag("RADINTAKE_OFFLINE", bool(runtime.get("offline", False)))
    forbid = env_flag("RADINTAKE_FORBID_MODEL_CALLS", bool(runtime.get("forbid_model_calls", False)))
    data_dir = Path(os.environ.get("RADINTAKE_DATA_DIR") or paths.get("data_dir") or "data")
    trace_env = os.environ.get("RADINTAKE_TRACE_DIR") or paths.get("trace_dir")
    trace_dir = Path(trace_env) if trace_env else None

    models: dict[str, ModelSpec] = {}
    for name, spec in (raw.get("models") or {}).items():
        if not isinstance(spec, dict) or "id" not in spec:
            raise ValueError(f"models.{name} needs an 'id'")
        kind = str(spec.get("kind", "local"))
        if kind not in ("local", "api"):
            raise ValueError(f"models.{name}.kind must be 'local' or 'api'")
        models[name] = ModelSpec(name=name, id=str(spec["id"]), kind=kind)

    return Config(
        offline=offline,
        forbid_model_calls=forbid,
        data_dir=data_dir,
        trace_dir=trace_dir,
        budgets=Budgets(
            max_steps=int(b["max_steps"]),
            stage_timeout_s=float(b["stage_timeout_s"]),
            retries=int(b["retries"]),
            backoff_s=float(b["backoff_s"]),
            case_deadline_s=float(b["case_deadline_s"]),
        ),
        pdf_dpi=int(raw["loader"]["pdf_dpi"]),
        seed=int(raw.get("seed", 140)),
        log_level=str(raw["logging"].get("level", "INFO")),
        models=models,
        raw=raw,
    )
