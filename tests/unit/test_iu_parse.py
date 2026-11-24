"""IU report XML parser on a hand-written fixture (synthetic text, no licensed IU content)."""

from pathlib import Path

from data_prep.iu_parse import counts, parse_dir, parse_report, write_jsonl

FIXTURE = b"""<?xml version="1.0" encoding="utf-8"?>
<eCitation>
  <uId id="CXR9001"/>
  <IUXRId id="9001"/>
  <MedlineCitation><Article><Abstract>
    <AbstractText Label="COMPARISON">Prior film from XXXX.</AbstractText>
    <AbstractText Label="INDICATION">Synthetic cough,
        for testing only</AbstractText>
    <AbstractText Label="FINDINGS">Synthetic findings line one. Line two.</AbstractText>
    <AbstractText Label="IMPRESSION">Synthetic impression.</AbstractText>
  </Abstract></Article></MedlineCitation>
  <MeSH>
    <major>Opacity/lung/base/left</major>
    <major>Cardiomegaly/mild</major>
    <minor>Calcinosis</minor>
  </MeSH>
  <parentImage id="CXR9001_IM-0001-1001"><figureId>F1</figureId></parentImage>
  <parentImage id="CXR9001_IM-0001-2001"><figureId>F2</figureId></parentImage>
</eCitation>
"""

EMPTY_SECTIONS = b"""<eCitation><uId id="CXR9002"/><IUXRId id="9002"/>
  <Abstract><AbstractText Label="FINDINGS"/><AbstractText Label="IMPRESSION">Normal.</AbstractText></Abstract>
  <MeSH><major>normal</major></MeSH>
</eCitation>"""


def test_parse_report_all_fields():
    r = parse_report(FIXTURE)
    assert r["report_id"] == "CXR9001"
    assert r["iu_id"] == 9001
    assert r["indication"] == "Synthetic cough, for testing only"  # whitespace normalised
    assert r["comparison"] == "Prior film from XXXX."
    assert r["findings"] == "Synthetic findings line one. Line two."
    assert r["impression"] == "Synthetic impression."
    assert r["mesh_major"] == ["Opacity/lung/base/left", "Cardiomegaly/mild"]
    assert r["mesh_minor"] == ["Calcinosis"]
    assert r["image_ids"] == ["CXR9001_IM-0001-1001", "CXR9001_IM-0001-2001"]


def test_missing_and_empty_sections_are_none():
    r = parse_report(EMPTY_SECTIONS)
    assert r["indication"] is None and r["comparison"] is None
    assert r["findings"] is None
    assert r["impression"] == "Normal."
    assert r["mesh_major"] == ["normal"] and r["mesh_minor"] == []
    assert r["image_ids"] == []


def test_parse_dir_sorted_and_jsonl(tmp_path: Path):
    d = tmp_path / "ecgen-radiology"
    d.mkdir()
    (d / "9002.xml").write_bytes(EMPTY_SECTIONS)
    (d / "10.xml").write_bytes(FIXTURE)
    records = parse_dir(tmp_path)
    assert [r["file"] for r in records] == ["10.xml", "9002.xml"]  # numeric, not lexical, order
    out = tmp_path / "reports.jsonl"
    write_jsonl(records, out)
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2
    c = counts(records)
    assert c["reports"] == 2 and c["with_all_four_sections"] == 1 and c["image_ids"] == 2
