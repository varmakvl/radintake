"""Integrity of the committed split / selection files, and of the generated corpus when present.

The committed checks (split, golden lock, DICOM selection) always run. Corpus checks need the
gitignored ``data/corpus`` (licensed IU text) and skip in CI.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from data_prep.split import golden_hash

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "splits"
CORPUS = ROOT / "data" / "corpus"
# golden v1 was scored once; its lock must never change (golden v2 is a separate list)
GOLDEN_V1_LOCK = "7b33866975ce4c34b0148875349634262aa28ba06787f1d78d0f399954836aa2"


def _ids(name: str) -> dict[str, str]:
    rows = (SPLITS / name).read_text(encoding="utf-8").splitlines()
    return dict(r.split("\t") for r in rows if r.strip())


def test_golden_list_is_hash_locked():
    assert golden_hash(SPLITS / "golden_ids.txt") == (SPLITS / "golden.lock").read_text().strip()


def test_dev_and_golden_are_disjoint_and_cover_the_corpus():
    dev, gold = _ids("dev_ids.txt"), _ids("golden_ids.txt")
    assert not set(dev) & set(gold)
    assert not set(dev.values()) & set(gold.values())  # split by report
    assert len(dev) + len(gold) >= 300 and len(gold) >= 100


def test_no_duplicate_group_crosses_the_split():
    dev, gold = set(_ids("dev_ids.txt").values()), set(_ids("golden_ids.txt").values())
    groups = json.loads((SPLITS / "dup_groups.json").read_text(encoding="utf-8"))["groups"]
    assert groups, "expected near-duplicate groups in IU"
    for g in groups:
        sides = {"dev" if r in dev else "golden" if r in gold else "?" for r in g}
        assert len(sides) == 1, f"dup group crosses the split: {g}"


def test_dicom_selection_meets_the_brief_and_carries_no_values():
    rows = list(csv.DictReader((SPLITS / "dicom_selection.csv").read_text(encoding="utf-8").splitlines()))
    assert len(rows) >= 40
    assert len({r["modality"] for r in rows}) >= 4
    assert any(int(r["burned_in_boxes"]) > 0 for r in rows)
    assert set(rows[0]) == {"dicom_id", "modality", "frames", "sha256", "phi_tags", "uid_remapped",
                            "private_phi_tags", "burned_in_boxes", "pixel_status"}


@pytest.mark.skipif(not (CORPUS / "manifest.csv").is_file(), reason="generated corpus not present (licensed data)")
def test_corpus_meets_the_brief():
    man = list(csv.DictReader((CORPUS / "manifest.csv").read_text(encoding="utf-8").splitlines()))
    assert len(man) >= 300
    assert {m["format"] for m in man} == {"png", "jpg", "pdf", "tiff"}
    assert {m["tier"] for m in man} == {"clean_scan", "poor_scan", "phone_photo"}
    assert any(int(m["n_pages"]) > 1 for m in man)
    assert sum(int(m["injected"]) for m in man) >= 10
    pages = hw_pages = 0
    rots = set()
    for m in man:
        gt = json.loads((CORPUS / "gt" / f"{m['doc_id']}.json").read_text(encoding="utf-8"))
        for p in gt["pages"]:
            pages += 1
            hw_pages += any(i["handwritten"] for i in p["instances"])
            w, h = p["size"]
            rots |= {o.get("deg") for o in p["degrade"].get("ops", []) if o.get("op") == "rotate90"}
            for i in p["instances"]:
                x0, y0, x1, y1 = i["bbox"]
                assert 0 <= x0 < x1 <= w + 1 and 0 <= y0 < y1 <= h + 1, (m["doc_id"], i["field"])
    assert hw_pages / pages >= 0.30
    assert {90, 180} <= rots


# ------------------------------------------------------------------------------- golden v2 (held-out set)
def test_golden_v2_list_is_hash_locked_and_v1_lock_is_unchanged():
    assert golden_hash(SPLITS / "golden_v2_ids.txt") == (SPLITS / "golden_v2.lock").read_text().strip()
    assert (SPLITS / "golden.lock").read_text().strip() == GOLDEN_V1_LOCK


def test_golden_v2_lock_detects_an_edited_id(tmp_path):
    lines = (SPLITS / "golden_v2_ids.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    doc, rep = lines[0].rstrip("\n").split("\t")
    lines[0] = f"{doc}\t{rep}9\n"
    edited = tmp_path / "golden_v2_ids.txt"
    edited.write_text("".join(lines), encoding="utf-8", newline="\n")
    assert golden_hash(edited) != (SPLITS / "golden_v2.lock").read_text().strip()


def test_golden_v2_is_disjoint_from_dev_and_golden_v1():
    dev, gold, v2 = _ids("dev_ids.txt"), _ids("golden_ids.txt"), _ids("golden_v2_ids.txt")
    assert len(v2) >= 100
    assert not set(v2) & (set(dev) | set(gold))                   # doc ids
    assert not set(v2.values()) & (set(dev.values()) | set(gold.values()))  # report ids
    assert len(set(v2.values())) == len(v2)                       # one doc per report


def test_no_duplicate_group_spans_v1_and_golden_v2():
    v1 = set(_ids("dev_ids.txt").values()) | set(_ids("golden_ids.txt").values())
    v2 = set(_ids("golden_v2_ids.txt").values())
    for g in json.loads((SPLITS / "dup_groups.json").read_text(encoding="utf-8"))["groups"]:
        assert not set(g) & v2, f"v1 dup group reaches golden v2: {g}"
    rec = json.loads((SPLITS / "golden_v2_dedup.json").read_text(encoding="utf-8"))
    assert rec["dropped_v1_dup_group"] > 0 and rec["seed"] != 140
    for g in rec["groups"]:
        assert set(g) <= v2 and not set(g) & v1, f"v2 dup group touches v1: {g}"
