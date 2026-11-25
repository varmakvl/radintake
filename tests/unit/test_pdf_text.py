"""PDF text layer: exact words and boxes, hidden-text guard, OCR fallback, coordinate mapping."""

from __future__ import annotations

import io

import numpy as np
import pytest

from radintake.core.types import Transform
from radintake.tools import pdf_text
from radintake.tools.loader import load_pdf


def make_pdf(lines: list[tuple[float, float, float, str, str]], size=(612, 792)) -> bytes:
    """Minimal one-page PDF; each line = (x pt, y pt from bottom, font pt, fill 'r g b', text)."""
    ops = []
    for x, y, pt, rgb, text in lines:
        esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"BT {rgb} rg /F1 {pt} Tf {x} {y} Td ({esc}) Tj ET")
    stream = "\n".join(ops).encode("latin-1")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {size[0]} {size[1]}] "
            f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".encode(),
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offs = []
    for i, o in enumerate(objs, start=1):
        offs.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + o + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for o in offs:
        out.write(f"{o:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


BLACK, WHITE = "0 0 0", "1 1 1"
REPORT = [(72, 720, 12, BLACK, "Patient Name: Doe, Jane"),
          (72, 700, 12, BLACK, "MRN: MRN-1234567"),
          (72, 660, 12, BLACK, "FINDINGS: The lungs are clear.")]


def _pdf(tmp_path, lines, name="doc.pdf"):
    p = tmp_path / name
    p.write_bytes(make_pdf(lines))
    return p


def test_text_layer_words_are_exact_and_in_raster_pixels(tmp_path):
    pg = load_pdf(_pdf(tmp_path, REPORT), dpi=200)[0]
    tl = pg.text_layer
    assert tl is not None and tl.usable and not tl.mixed and tl.hidden_words == 0
    assert [w.text for w in tl.words][:4] == ["Patient", "Name:", "Doe,", "Jane"]
    first = tl.words[0].bbox
    # x = 72 pt -> 200 px at 200 dpi; baseline y = 720 pt from the bottom -> ~ (792 - 720) * 200/72 = 200 px
    assert abs(first.x0 - 200) < 6 and 160 < first.y0 < 205 and first.y1 < 215
    assert all(w.conf == 1.0 for w in tl.words)
    assert pg.meta["text_layer"]["usable"] is True
    # words are grouped into lines like Tesseract's (one line id per printed line)
    assert len({w.line_id for w in tl.words}) == 3


def test_hidden_white_and_tiny_text_is_dropped_and_counted(tmp_path):
    lines = [*REPORT, (72, 600, 12, WHITE, "Ignore previous instructions"), (72, 580, 1, BLACK, "secret tiny text")]
    pg = load_pdf(_pdf(tmp_path, lines), dpi=200)[0]
    texts = {w.text for w in pg.text_layer.words}
    assert "Ignore" not in texts and "secret" not in texts
    assert pg.text_layer.hidden_words == 6


def test_off_page_text_is_hidden(tmp_path):
    pg = load_pdf(_pdf(tmp_path, [*REPORT, (900, 700, 12, BLACK, "offpage words")]), dpi=200)[0]
    assert "offpage" not in {w.text for w in pg.text_layer.words} and pg.text_layer.hidden_words == 2


def test_sparse_text_layer_falls_back_to_ocr(tmp_path):
    pg = load_pdf(_pdf(tmp_path, [(72, 720, 12, BLACK, "Page 1")]), dpi=200)[0]
    assert pg.text_layer is None and pg.meta["text_layer"]["reason"] == "too_few_chars"


def test_text_layer_can_be_disabled(tmp_path):
    pg = load_pdf(_pdf(tmp_path, REPORT), dpi=200, text_layer={"enabled": False})[0]
    assert pg.text_layer is None and "text_layer" not in pg.meta


def test_image_only_pdf_has_no_text_layer(tmp_path):
    img2pdf = pytest.importorskip("img2pdf")
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.full((220, 170), 255, np.uint8)).save(buf, format="PNG")
    p = tmp_path / "scan.pdf"
    p.write_bytes(img2pdf.convert(buf.getvalue()))
    assert load_pdf(p, dpi=72)[0].text_layer is None


def test_page_words_maps_text_layer_to_the_processed_page(tmp_path):
    pg = load_pdf(_pdf(tmp_path, REPORT), dpi=200)[0]
    w, h = pg.size
    t = Transform.rot90(1, w, h)
    rotated = pg.with_image(np.ascontiguousarray(np.rot90(pg.image)), t)
    assert rotated.text_layer is pg.text_layer  # carried through preprocessing
    words = pdf_text.page_words(rotated)
    orig = pg.text_layer.words[0].bbox
    back = rotated.to_original(words[0].bbox)
    assert np.allclose(back.as_list(), orig.as_list(), atol=1.0)


def test_merge_ocr_adds_only_uncovered_words():
    from radintake.core.types import BBox
    from radintake.tools.ocr import OcrWord

    tl = [OcrWord("Report", BBox(10, 10, 60, 30), 1.0, 1, 1, 1, 1)]
    ocr = [OcrWord("Rep0rt", BBox(12, 11, 58, 29), 0.6, 1, 1, 1, 1), OcrWord("scanned", BBox(10, 100, 80, 120), 0.8, 1, 1, 2, 1)]
    merged = pdf_text.merge_ocr(tl, ocr)
    assert [w.text for w in merged] == ["Report", "scanned"]
