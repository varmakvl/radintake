"""Parse IU Open-i report XML files into data/iu/reports.jsonl.

    python -m data_prep.iu_parse [--reports-dir data/iu/reports] [--out data/iu/reports.jsonl]

One JSON object per report, sorted by numeric report id::

    {"report_id": "CXR1", "iu_id": 1, "file": "1.xml",
     "indication": str|None, "comparison": str|None, "findings": str|None, "impression": str|None,
     "mesh_major": [...], "mesh_minor": [...], "image_ids": ["CXR1_1_IM-0001-3001", ...]}

A section is None when its <AbstractText> element is absent or empty. Output stays under the
gitignored data/ directory (licence CC BY-NC-ND).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SECTIONS = ("indication", "comparison", "findings", "impression")


def _text(el: ET.Element | None) -> str | None:
    if el is None:
        return None
    txt = " ".join("".join(el.itertext()).split())
    return txt or None


def parse_report(source: Path | str | bytes) -> dict:
    """Parse one report XML (path or raw bytes/str) into a flat dict."""
    if isinstance(source, Path):
        root = ET.parse(source).getroot()
        fname = source.name
    else:
        root = ET.fromstring(source)
        fname = None
    uid = root.find("uId")
    iu = root.find("IUXRId")
    sections: dict[str, str | None] = dict.fromkeys(SECTIONS)
    for at in root.iter("AbstractText"):
        label = (at.get("Label") or "").strip().lower()
        if label in sections:
            sections[label] = _text(at)
    mesh = root.find("MeSH")
    major = [t for t in (_text(e) for e in mesh.findall("major"))] if mesh is not None else []
    minor = [t for t in (_text(e) for e in mesh.findall("minor"))] if mesh is not None else []
    iu_id = iu.get("id") if iu is not None else None
    return {
        "report_id": uid.get("id") if uid is not None else None,
        "iu_id": int(iu_id) if iu_id and iu_id.isdigit() else iu_id,
        "file": fname,
        **sections,
        "mesh_major": [t for t in major if t],
        "mesh_minor": [t for t in minor if t],
        "image_ids": [p.get("id") for p in root.findall("parentImage") if p.get("id")],
    }


def _sort_key(path: Path):
    m = re.match(r"(\d+)", path.stem)
    return (int(m.group(1)) if m else sys.maxsize, path.name)


def parse_dir(reports_dir: Path) -> list[dict]:
    files = sorted(reports_dir.rglob("*.xml"), key=_sort_key)
    return [parse_report(f) for f in files]


def write_jsonl(records: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=False) + "\n")


def counts(records: list[dict]) -> dict:
    c = {"reports": len(records)}
    for s in SECTIONS:
        c[f"with_{s}"] = sum(1 for r in records if r[s])
    c["with_all_four_sections"] = sum(1 for r in records if all(r[s] for s in SECTIONS))
    c["with_mesh_major"] = sum(1 for r in records if r["mesh_major"])
    c["with_images"] = sum(1 for r in records if r["image_ids"])
    c["image_ids"] = sum(len(r["image_ids"]) for r in records)
    return c


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m data_prep.iu_parse", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports-dir", type=Path, default=Path("data/iu/reports"))
    ap.add_argument("--out", type=Path, default=Path("data/iu/reports.jsonl"))
    args = ap.parse_args(argv)
    if not args.reports_dir.is_dir():
        print(f"{args.reports_dir} not found: run `python -m data_prep.download iu` first")
        return 1
    records = parse_dir(args.reports_dir)
    write_jsonl(records, args.out)
    for k, v in counts(records).items():
        print(f"{k}: {v}")
    print(f"wrote {args.out}")
    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())
