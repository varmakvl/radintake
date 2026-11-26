"""radintake: PHI-safe multi-agent intake of radiology reports and DICOM.

Public API (stable under semver, see ``docs/api-stability.md``) is exactly ``__all__``::

    import radintake

    result = radintake.process("report.pdf", config=radintake.RadintakeConfig(offline=True))
    result.status, result.report.impression.value

Every other module (``radintake.pipeline``, ``radintake.agents.*``, ``radintake.api.*`` ...) is
internal and may change in any release. Names are imported lazily so ``import radintake`` stays
light (no FastAPI, uvicorn or fastembed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_SDK = frozenset({"Pipeline", "RadintakeConfig", "BudgetConfig", "ModelConfig", "process",
                  "aprocess", "RadintakeError", "UnsupportedInputError", "ConfigError"})
_RESULTS = frozenset({"Result", "Report", "ReportField", "Provenance", "Phi", "Detection", "Dicom",
                      "Table", "TableCell"})

__all__ = [
    "BudgetConfig",
    "ConfigError",
    "Detection",
    "Dicom",
    "ModelConfig",
    "Phi",
    "Pipeline",
    "Provenance",
    "RadintakeConfig",
    "RadintakeError",
    "Report",
    "ReportField",
    "Result",
    "Table",
    "TableCell",
    "UnsupportedInputError",
    "__version__",
    "aprocess",
    "process",
]

if TYPE_CHECKING:  # pragma: no cover
    from radintake.results import (
        Detection,
        Dicom,
        Phi,
        Provenance,
        Report,
        ReportField,
        Result,
        Table,
        TableCell,
    )
    from radintake.sdk import (
        BudgetConfig,
        ConfigError,
        ModelConfig,
        Pipeline,
        RadintakeConfig,
        RadintakeError,
        UnsupportedInputError,
        aprocess,
        process,
    )


def __getattr__(name: str) -> Any:
    if name in _SDK:
        from radintake import sdk

        return getattr(sdk, name)
    if name in _RESULTS:
        from radintake import results

        return getattr(results, name)
    raise AttributeError(f"module 'radintake' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
