"""Pydantic mirror of ``schemas/report.v1.json`` (the ``report`` block of the CLI output).

The JSON Schema is the contract; this model is how Python code builds and checks reports.
``Report.to_dict()`` emits exactly the schema shape (optional keys only when set), and
``tests/unit/test_extract_ocr.py`` checks the two stay in step.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

REPORT_SCHEMA_VERSION = "report.v1"
HEADER_FIELDS = ("patient_name", "patient_id", "date_of_birth", "accession_number")
EXAM_FIELDS = ("exam_date", "modality")
SECTION_FIELDS = ("indication", "comparison", "findings", "impression")
ALL_FIELDS = HEADER_FIELDS + EXAM_FIELDS + SECTION_FIELDS
REQUIRED_FIELDS = HEADER_FIELDS + SECTION_FIELDS

Source = Literal["ocr", "vlm", "hybrid"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Provenance(_Strict):
    page: int = Field(ge=1)
    bbox: list[float] = Field(min_length=4, max_length=4)
    source: Source | None = None

    @field_validator("bbox")
    @classmethod
    def _ordered(cls, v: list[float]) -> list[float]:
        if not (v[0] <= v[2] and v[1] <= v[3]):
            raise ValueError(f"bbox must be [x0, y0, x1, y1] with x0<=x1, y0<=y1: {v}")
        return [round(float(x), 2) for x in v]


class UntrustedText(_Strict):
    text: str
    reason: str | None = None
    provenance: list[Provenance] = Field(default_factory=list)


class ReportField(_Strict):
    value: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance: list[Provenance] = Field(default_factory=list)
    low_confidence: bool | None = None
    grounded: bool | None = None
    needs_review: bool | None = None
    untrusted_text: list[UntrustedText] | None = None

    @classmethod
    def missing(cls) -> ReportField:
        return cls(value=None, confidence=0.0, provenance=[])


class Report(_Strict):
    patient_name: ReportField = Field(default_factory=ReportField.missing)
    patient_id: ReportField = Field(default_factory=ReportField.missing)
    date_of_birth: ReportField = Field(default_factory=ReportField.missing)
    accession_number: ReportField = Field(default_factory=ReportField.missing)
    exam_date: ReportField | None = None
    modality: ReportField | None = None
    indication: ReportField = Field(default_factory=ReportField.missing)
    comparison: ReportField = Field(default_factory=ReportField.missing)
    findings: ReportField = Field(default_factory=ReportField.missing)
    impression: ReportField = Field(default_factory=ReportField.missing)

    def to_dict(self) -> dict[str, Any]:
        """Schema-shaped dict: ``value`` is always present (possibly null), other optional keys
        only when set."""
        out: dict[str, Any] = {}
        for name in ALL_FIELDS:
            f: ReportField | None = getattr(self, name)
            if f is None:
                continue
            d = f.model_dump(exclude_none=True)
            d["value"] = f.value
            d["provenance"] = [p.model_dump(exclude_none=True) for p in f.provenance]
            if f.untrusted_text is not None:
                d["untrusted_text"] = [
                    {**u.model_dump(exclude_none=True),
                     "provenance": [p.model_dump(exclude_none=True) for p in u.provenance]}
                    for u in f.untrusted_text]
            out[name] = d
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Report:
        return cls.model_validate(data)
