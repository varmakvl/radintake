"""Typed results of the public SDK (:class:`radintake.sdk.Pipeline`).

Stability: the models re-exported in ``radintake.__all__`` are public (``docs/api-stability.md``);
``BBox``, ``REPORT_FIELDS`` and the ``_Model`` base are internal.

:class:`Result` mirrors ``schemas/cli_output.v1.json`` field for field (plus the optional
``tables`` array of ``schemas/tables.v1.json``), so ``result.model_dump(mode="json")`` is a
valid ``cli_output.v1`` document - the same document the batch CLI writes to disk.

Serialisation rules that keep the dump schema-valid:

* keys that are *nullable* in the schema (``error``, ``report``, ``phi``, ``dicom``,
  ``field.value``, ``table.section``) are always dumped, ``None`` included;
* optional keys that are *not* nullable (``field.grounded``, ``provenance.source``,
  ``report.exam_date``, ``detection.bbox``, ``tables`` ...) are left out when ``None``;
* unknown keys are kept (``extra="allow"``) so newer pipeline output round-trips unchanged
  (e.g. ``field.untrusted_text``);
* the in-memory artefacts (:attr:`Result.masked_pages`, :attr:`Result.dicom_bytes`) are
  ``exclude=True`` fields: they are never part of a dump.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer

BBox = list[float]
InputType = Literal["dicom", "scan", "photo", "pdf", "tiff", "unknown"]
Status = Literal["ok", "needs_review", "error"]
Flag = Literal["low_confidence", "ungrounded_value", "omission", "schema_error",
               "prompt_injection_suspected", "phi_uncertain", "preprocess_failed", "timeout"]

REPORT_FIELDS = ("patient_name", "patient_id", "date_of_birth", "accession_number", "exam_date",
                 "modality", "indication", "comparison", "findings", "impression")


class _Model(BaseModel):
    """Base: extra keys kept, ``None`` dropped for keys not in ``_nullable``."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)
    _nullable: ClassVar[frozenset[str]] = frozenset()

    @model_serializer(mode="wrap")
    def _drop_absent(self, handler: Any) -> Any:
        data = handler(self)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None or k in self._nullable}
        return data


class Provenance(_Model):
    page: int = Field(ge=1)
    bbox: BBox
    source: Literal["ocr", "vlm", "hybrid"] | None = None


class ReportField(_Model):
    _nullable = frozenset({"value"})

    value: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: list[Provenance] = Field(default_factory=list)
    grounded: bool | None = None
    needs_review: bool | None = None


class Report(_Model):
    patient_name: ReportField
    patient_id: ReportField
    date_of_birth: ReportField
    accession_number: ReportField
    indication: ReportField
    comparison: ReportField
    findings: ReportField
    impression: ReportField
    exam_date: ReportField | None = None
    modality: ReportField | None = None

    def values(self) -> dict[str, str | None]:
        """``{field name: value}`` for every field present."""
        return {k: getattr(self, k).value for k in REPORT_FIELDS if getattr(self, k, None) is not None}


class Detection(_Model):
    type: str
    page: int = Field(ge=1)
    bbox: BBox | None = None
    dicom_tag: str | None = None


class Phi(_Model):
    masked_artifacts: list[str] | None = None
    detections: list[Detection] | None = None


class Dicom(_Model):
    output_path: str | None = None
    valid: bool | None = None
    header_tags_deidentified: list[str] | None = None
    burned_in_regions: list[BBox] | None = None


class TableCell(_Model):
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    text: str
    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)


class Table(_Model):
    _nullable = frozenset({"section"})

    page: int = Field(ge=1)
    bbox: BBox
    n_rows: int = Field(ge=1)
    n_cols: int = Field(ge=1)
    method: Literal["ruled", "rules_only", "aligned"]
    text_source: Literal["pdf_text", "ocr"]
    confidence: float = Field(ge=0.0, le=1.0)
    section: str | None = None
    cells: list[TableCell] = Field(default_factory=list)


class Result(_Model):
    """One processed input - a ``cli_output.v1`` document with typed access.

    ``masked_pages`` (PNG bytes of each PHI-masked page, in ``phi.masked_artifacts`` order) and
    ``dicom_bytes`` (the de-identified DICOM) hold the artefacts in memory; they are excluded
    from every dump. ``phi.masked_artifacts`` / ``dicom.output_path`` are names relative to the
    ``output_dir`` - files only exist there when an ``output_dir`` was given.
    """

    _nullable = frozenset({"error", "report", "phi", "dicom"})

    schema_version: Literal["cli_output.v1"] = "cli_output.v1"
    input: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_type: InputType
    status: Status
    trace_id: str = Field(min_length=1)
    error: str | None = None
    n_pages: int = Field(default=0, ge=0)
    report: Report | None = None
    phi: Phi | None = None
    dicom: Dicom | None = None
    flags: list[Flag] = Field(default_factory=list)
    tables: list[Table] | None = None

    masked_pages: list[bytes] = Field(default_factory=list, exclude=True, repr=False)
    dicom_bytes: bytes | None = Field(default=None, exclude=True, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_document(self) -> dict[str, Any]:
        """The ``cli_output.v1`` JSON document (what the batch CLI writes)."""
        return self.model_dump(mode="json")


__all__ = ["BBox", "Detection", "Dicom", "Phi", "Provenance", "Report", "ReportField", "Result",
           "Table", "TableCell"]
