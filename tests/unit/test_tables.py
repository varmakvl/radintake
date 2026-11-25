"""Table detection (ruled / rules-only / aligned), cell re-read, extraction and output contract."""

from __future__ import annotations

import cv2
import numpy as np

from radintake.core.types import BBox, Page
from radintake.tools.ocr import OcrWord
from radintake.tools.tables import detect_tables

W, H = 1700, 1200
CELL_W, CELL_H = 220, 50
ROWS = [["Lesion", "Location", "Size"], ["1", "RUL", "9"], ["2", "LLL", "4"], ["3", "Lingula", "5"]]


def _word(text, x, y, line, word=1, conf=0.95):
    return OcrWord(text, BBox(x, y, x + 18 * len(text), y + 28), conf, 1, 1, line, word)


def _table_words(x0=200, y0=300, rows=ROWS, line0=10):
    ws = []
    for r, row in enumerate(rows):
        for c, txt in enumerate(row):
            ws.append(_word(txt, x0 + c * CELL_W + 10, y0 + r * CELL_H + 11, line0 + r, c + 1))
    return ws


def _page(grid=True, rules=False, x0=200, y0=300, rows=ROWS):
    img = np.full((H, W), 255, np.uint8)
    nr, nc = len(rows), len(rows[0])
    x1, y1 = x0 + nc * CELL_W, y0 + nr * CELL_H
    if grid:
        for r in range(nr + 1):
            cv2.line(img, (x0, y0 + r * CELL_H), (x1, y0 + r * CELL_H), 0, 2)
        for c in range(nc + 1):
            cv2.line(img, (x0 + c * CELL_W, y0), (x0 + c * CELL_W, y1), 0, 2)
    if rules:
        for y in (y0, y0 + CELL_H, y1):
            cv2.line(img, (x0, y), (x1, y), 0, 2)
    return img


def test_ruled_grid_is_read_cell_by_cell():
    ts = detect_tables(_page(), _table_words())
    assert len(ts) == 1
    t = ts[0]
    assert (t.method, t.n_rows, t.n_cols) == ("ruled", 4, 3)
    assert t.rows() == ROWS
    assert t.confidence > 0.9


def test_rules_only_table_gets_columns_from_word_gaps():
    ts = detect_tables(_page(grid=False, rules=True), _table_words())
    assert len(ts) == 1 and ts[0].method == "rules_only" and ts[0].rows() == ROWS


def test_borderless_aligned_table():
    ts = detect_tables(_page(grid=False), _table_words())
    assert len(ts) == 1 and ts[0].method == "aligned" and ts[0].rows() == ROWS


def test_prose_and_key_value_header_are_not_tables():
    img = np.full((H, W), 255, np.uint8)
    cv2.line(img, (100, 100), (1600, 100), 0, 2)  # page rules above and below prose
    cv2.line(img, (100, 400), (1600, 400), 0, 2)
    words = []
    for li in range(5):
        x = 120
        for wi, t in enumerate(("The", "lungs", "are", "clear", "without", "focal", "consolidation.")):
            words.append(_word(t, x + (li % 2) * 40, 130 + li * 45, li + 1, wi + 1))
            x += 18 * len(t) + 14
    words += [_word("Patient", 120, 500, 20, 1), _word("Name:", 260, 500, 20, 2), _word("Doe,", 500, 500, 20, 3)]
    assert detect_tables(img, words) == []


def test_scratch_line_through_the_page_does_not_split_or_extend_a_table():
    img = _page()
    cv2.line(img, (1100, 0), (1100, H - 1), 0, 2)  # scanner scratch crossing everything
    ts = detect_tables(img, _table_words())
    assert len(ts) == 1 and ts[0].n_cols == 3


def test_low_confidence_aligned_tables_are_dropped():
    ws = [OcrWord(w.text, w.bbox, 0.2, w.block, w.par, w.line, w.word) for w in _table_words()]
    assert detect_tables(_page(grid=False), ws) == []


def test_cell_reread_replaces_garbled_cell_text():
    ws = _table_words()
    ws = [OcrWord("|%#" if w.text == "Location" else w.text, w.bbox, 0.3 if w.text == "Location" else w.conf,
                  w.block, w.par, w.line, w.word) for w in ws]

    def fake_ocr(img, psm=7):
        assert psm == 7 and img.ndim == 2
        return [OcrWord("Location", BBox(0, 0, 10, 10), 0.97, 1, 1, 1, 1)]

    t = detect_tables(_page(), ws, ocr_fn=fake_ocr)[0]
    assert t.rows()[0][1] == "Location"


def test_apply_cells_and_output_boxes_are_in_original_coordinates():
    from radintake.agents.tables import apply_cells, table_output
    from radintake.core.types import Transform

    t = detect_tables(_page(), _table_words())[0]
    t.cells[1].text = "Site"  # as if re-read
    words = apply_cells(_table_words(), [t], {(0, 0, 1)})
    texts = [w.text for w in words]
    assert "Site" in texts and "Location" not in texts
    page = Page(index=1, image=_page(), original_size=(W, H)).with_image(_page(), Transform.translation(10, 0))
    out = table_output(t, page)
    assert out["cells"][0]["bbox"][0] == round(t.cells[0].bbox.x0 - 10, 2)
    assert out["text_source"] == "ocr" and out["n_rows"] == 4


def test_extraction_keeps_table_text_out_of_findings_and_reads_exam_columns():
    from radintake.agents import extract_ocr as E

    words = [_word("FINDINGS:", 200, 150, 1), _word("Nodules", 200, 200, 2), _word("below.", 400, 200, 2, 2)]
    words += _table_words()
    exam_rows = [["Exam Date", "Modality", "Accession"], ["03/14/2026", "CT", "ACC-778899"]]
    ex_words = _table_words(y0=700, rows=exam_rows, line0=40)
    img = np.maximum(_page(), 0)
    img2 = _page(y0=700, rows=exam_rows)
    img = np.minimum(img, img2)
    allw = words + ex_words
    tables = detect_tables(img, allw)
    assert len(tables) == 2
    page = Page(index=1, image=img, original_size=(W, H))
    res = E.extract_pages([page], {1: allw}, tables={1: tables})
    rep = res.report
    assert rep.findings.value == "Nodules below."
    assert rep.exam_date.value == "03/14/2026" and rep.modality.value == "CT"
    assert rep.accession_number.value == "ACC-778899"
    assert res.meta["table_sections"] == {"1:0": "findings"}


def test_tables_schema_and_v1_output_contract():
    """Tables validate against tables.v1; a document carrying them still validates against cli_output.v1."""
    import json

    from jsonschema import Draft202012Validator

    from radintake.agents.tables import table_output
    from radintake.config import REPO_ROOT

    t = detect_tables(_page(), _table_words())[0]
    out = table_output(t, Page(index=1, image=_page(), original_size=(W, H)))
    tschema = json.loads((REPO_ROOT / "schemas" / "tables.v1.json").read_text(encoding="utf-8"))
    assert not list(Draft202012Validator(tschema).iter_errors([out]))
    v1 = json.loads((REPO_ROOT / "schemas" / "cli_output.v1.json").read_text(encoding="utf-8"))
    doc = {"schema_version": "cli_output.v1", "input": "a.pdf", "input_sha256": "0" * 64, "input_type": "pdf",
           "status": "needs_review", "trace_id": "t", "flags": ["prompt_injection_suspected"], "tables": [out]}
    assert not list(Draft202012Validator(v1).iter_errors(doc))


def test_boxed_columns_single_band_split_into_label_and_value_rows():
    """Each column drawn as its own box with label above value (one band): two rows, not a miss."""
    from radintake.agents import extract_ocr as E

    img = np.full((H, W), 255, np.uint8)
    x0, y0, cw = 120, 500, 400
    for i in range(3):
        cv2.rectangle(img, (x0 + i * cw, y0), (x0 + (i + 1) * cw, y0 + 120), 0, 3)
    rows = [["Exam", "Exam Date", "Modality"], ["XR CHEST", "2024-06-11", "CR"]]
    words = [_word("Accession", 120, 420, 1), _word("No.:", 300, 420, 1, 2), _word("ACC-55019384", 480, 420, 1, 3)]
    for r, row in enumerate(rows):
        for c, txt in enumerate(row):
            for k, part in enumerate(txt.split()):
                words.append(_word(part, x0 + c * cw + 15 + 120 * k, y0 + 12 + 50 * r, 10 + r, 10 * c + k + 1))
    ts = detect_tables(img, words)
    assert len(ts) == 1 and ts[0].method == "ruled" and ts[0].rows() == rows
    page = Page(index=1, image=img, original_size=(W, H))
    rep = E.extract_pages([page], {1: words}, tables={1: ts}).report
    assert rep.accession_number.value == "ACC-55019384"  # the labelled header line, not a table column
    assert rep.exam_date.value == "2024-06-11" and rep.modality.value == "CR"
