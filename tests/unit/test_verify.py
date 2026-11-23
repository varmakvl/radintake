"""Verifier agent: grounding, omission, schema, cross-field checks; never changes a value."""

from __future__ import annotations

import copy
import random
from typing import ClassVar

import numpy as np
import pytest

from radintake.agents.extract_ocr import OcrFirstExtractor, extract_pages, page_lines
from radintake.agents.verify import PageOcr, load_params, verify_report
from radintake.core.types import BBox, Page, Transform
from radintake.tools.ocr import OcrWord

ROWS = [
    (200, ["Patient", "Name:", "Marlowe", "Quistgaard"]),
    (260, ["Patient", "ID:", "MRN-7730215"]),
    (320, ["Date", "of", "Birth:", "1961-04-17"]),
    (380, ["Accession", "No.:", "ACC-55019384"]),
    (640, ["INDICATION:"]),
    (690, ["Shortness", "of", "breath."]),
    (780, ["COMPARISON:", "None."]),
    (870, ["FINDINGS:"]),
    (920, ["The", "heart", "size", "is", "normal."]),
    (1060, ["IMPRESSION:"]),
    (1110, ["No", "acute", "disease."]),
]


def _words(rows, x0=100, conf=0.95):
    out = []
    for ln, (y, toks) in enumerate(rows, start=1):
        x = x0
        for wi, t in enumerate(toks, start=1):
            w = 14 * len(t)
            out.append(OcrWord(t, BBox(x, y, x + w, y + 28), conf, 1, 1, ln, wi))
            x += w + 14
    return out


def _page():
    return Page(index=1, image=np.full((2200, 1700), 255, np.uint8), original_size=(1700, 2200),
                transform=Transform.identity())


def _setup(rows=ROWS):
    words = _words(rows)
    res = extract_pages([_page()], {1: words})
    ex = OcrFirstExtractor()
    po = PageOcr([(w.bbox.as_list(), w.text) for w in words],
                 {h[0] for ln in page_lines(1, words) if (h := ex.heading_of(ln))})
    return res.report.to_dict(), {1: po}


def test_clean_report_is_ok_and_every_value_grounded():
    report, ocr = _setup()
    res = verify_report(report, ocr)
    assert res.status == "ok", (res.flags, res.reasons)
    assert not res.flags and not res.review_fields
    for f in res.report.values():
        if f["value"]:
            assert f["grounded"] is True
            assert not f.get("needs_review")


def test_hallucinated_value_is_ungrounded_and_kept():
    report, ocr = _setup()
    report["findings"]["value"] = "Large right pleural effusion with mediastinal shift."
    res = verify_report(report, ocr)
    f = res.report["findings"]
    assert f["grounded"] is False and f["needs_review"] is True
    assert f["value"] == "Large right pleural effusion with mediastinal shift."  # kept, never replaced
    assert "ungrounded_value" in res.flags
    assert res.status == "needs_review" and res.review_fields == ["findings"]


def test_empty_field_with_heading_is_omission():
    rows = [r for r in ROWS if r[0] not in (1110,)]  # impression heading stays, body gone
    report, ocr = _setup(rows)
    assert report["impression"]["value"] is None
    res = verify_report(report, ocr)
    assert "omission" in res.flags
    assert res.report["impression"]["needs_review"] is True
    assert res.status == "needs_review"


def test_absent_heading_and_empty_field_is_not_omission():
    rows = [r for r in ROWS if r[0] not in (780,)]  # no COMPARISON heading at all
    report, ocr = _setup(rows)
    res = verify_report(report, ocr)
    assert "omission" not in res.flags


def test_malformed_report_is_schema_error():
    report, ocr = _setup()
    del report["findings"]["provenance"]
    res = verify_report(report, ocr)
    assert "schema_error" in res.flags and res.status == "needs_review"
    assert verify_report(None, ocr).flags == {"schema_error"}


def test_low_confidence_flagged():
    report, ocr = _setup()
    report["indication"]["confidence"] = 0.2
    res = verify_report(report, ocr, params={"low_conf_threshold": 0.5})
    assert "low_confidence" in res.flags
    assert res.report["indication"]["needs_review"] is True and res.report["indication"]["low_confidence"] is True


def test_cross_field_checks():
    report, ocr = _setup()
    report["date_of_birth"]["value"] = "1961-13-45"
    report["patient_id"]["value"] = "??"
    report["impression"]["value"] = report["findings"]["value"]
    res = verify_report(report, ocr)
    assert {"invalid_date:date_of_birth", "invalid_id:patient_id", "impression_equals_findings"} <= set(res.reasons)
    assert res.status == "needs_review"


def test_value_grounded_by_whole_page_when_box_is_wrong():
    report, ocr = _setup()
    report["indication"]["provenance"][0]["bbox"] = [1500.0, 2000.0, 1600.0, 2100.0]  # points elsewhere
    res = verify_report(report, ocr)
    assert res.report["indication"]["grounded"] is True


def test_load_params_merges_config():
    class C:
        raw: ClassVar = {"verify": {"ground_min_ratio": 70, "id_patterns": {"patient_id": "^X$"}}}

    p = load_params(C())
    assert p["ground_min_ratio"] == 70 and p["id_patterns"]["patient_id"] == "^X$"
    assert "accession_number" in p["id_patterns"]


def test_verifier_never_changes_values():
    """Property: output values == input values for random (mostly hallucinated) value overrides."""
    rng = random.Random(140)
    names = ["patient_name", "indication", "findings", "impression", "patient_id"]
    for _ in range(40):
        report, ocr = _setup()
        for k in rng.sample(names, rng.randint(1, len(names))):
            report[k]["value"] = "".join(rng.choice("abcXYZ 019-./") for _ in range(rng.randint(1, 40)))
        before = {k: copy.deepcopy(f["value"]) for k, f in report.items()}
        res = verify_report(report, ocr)
        assert {k: f["value"] for k, f in res.report.items()} == before
        assert {k: f["value"] for k, f in report.items()} == before  # input untouched too


@pytest.mark.skipif(not __import__("shutil").which("tesseract"), reason="tesseract not installed")
def test_report_clean_png_all_text_fields_grounded(tmp_path):
    from radintake.config import load_config
    from radintake.pipeline import process_file
    from tests.integration.fixtures import REPORTS, render_report

    p = tmp_path / "report_clean.png"
    render_report(REPORTS[0]).save(p)
    doc = process_file(p, tmp_path / "out", config=load_config())
    for name in ("indication", "comparison", "findings", "impression"):
        assert doc["report"][name]["grounded"] is True, name
