"""DICOM ground truth from the TCIA Pseudo-PHI evaluation set and its curated de-identified twin.

    python -m data_prep.dicom_gt [--tcia data/tcia] [--out data/dicom_gt] [--n 48] [--seed 140]

Derivation
----------
The download holds both halves of DOI 10.7937/s17z-r072: 21 *evaluation* patients (synthetic PHI
inserted) and their 21 TCIA-curated *de-identified* twins (``Pseudo-PHI-001..021``).
``Pseudo-PHI-DICOM-Dataset-uid_crosswalk.csv`` maps every evaluation UID to its de-identified UID,
so each evaluation instance is paired with its curated copy by SOPInstanceUID.

* **Header PHI** = every data element of the evaluation instance (sequences recursed, file meta and
  PixelData excluded) that the curator removed, emptied or changed. UIDs that were remapped are
  counted separately (``uid_remapped``) - they are identifiers, the expected action is "replace".
* **Burned-in text** = pixels that differ between the evaluation and curated instance (same frame
  shape), dilated and grouped into connected components -> boxes ``[x0, y0, x1, y1]`` per frame.

Where it may be incomplete (also written to DICOM_GT.md):

* the curated copy reflects TCIA's own de-identification policy - an element TCIA kept (e.g. a date
  it judged safe) is not labelled PHI here even if another policy would remove it;
* burned-in text the curator did NOT clean is invisible to a pixel diff;
* a curated image re-encoded or cropped (different shape) gives no burned-in labels;
* differences caused by lossy re-encoding would appear as noise boxes (none seen: tiny components are
  dropped below ``MIN_AREA``).

Selection: every pair with burned-in boxes first, then a seeded draw balanced across modalities and
patients up to ``n`` instances. Selected evaluation files are copied as ``inputs/DICOM_<nnn>.dcm`` so
no file or folder name carries the synthetic PHI the original paths contain.

Outputs: ``<out>/inputs/*.dcm`` + ``<out>/gt.jsonl`` (full GT incl. values; gitignored) and the
committed, value-free ``splits/dicom_selection.csv`` and ``data_prep/DICOM_GT.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 140
MIN_AREA = 12  # px; smaller diff components are treated as noise
# Image-pixel-module elements the curator rewrote while re-encoding pixels: they describe pixel layout,
# not identity, so a difference there is a curation artefact and not ground-truth PHI.
NOT_PHI = {"PlanarConfiguration", "PixelAspectRatio", "BitsStored", "HighBit", "PixelRepresentation",
           "PhotometricInterpretation", "SamplesPerPixel", "LossyImageCompression", "TransferSyntaxUID"}


def _walk(ds, prefix=""):
    """(path, element) for every element, sequences recursed (path like 'Seq[0].Keyword')."""
    for el in ds:
        if el.tag == 0x7FE00010:  # PixelData
            continue
        name = el.keyword or f"({el.tag.group:04X},{el.tag.element:04X})"
        path = f"{prefix}{name}"
        if el.VR == "SQ":
            for i, item in enumerate(el.value or []):
                yield from _walk(item, f"{path}[{i}].")
        else:
            yield path, el


def header_diff(ev, de) -> dict[str, list[dict]]:
    dmap = dict(_walk(de))
    out = {"removed": [], "changed": [], "emptied": [], "uid_remapped": []}
    for path, el in _walk(ev):
        val = el.value
        if val in (None, "", b"") or (hasattr(val, "__len__") and len(val) == 0):
            continue  # nothing to protect
        if path.split("[")[0].split(".")[-1] in NOT_PHI:
            continue
        rec = {"tag": path, "vr": el.VR, "private": bool(el.tag.is_private), "value": _str(val)}
        other = dmap.get(path)
        if other is None:
            out["removed"].append(rec)
        elif other.value in (None, "", b""):
            out["emptied"].append(rec)
        elif _str(other.value) != rec["value"]:
            (out["uid_remapped"] if el.VR == "UI" else out["changed"]).append(rec)
    return out


def _str(v) -> str:
    if isinstance(v, bytes):
        return f"<bytes:{len(v)}>"
    return str(v)[:200]


def burned_boxes(ev, de, masks: list | None = None) -> tuple[list[dict], str]:
    import cv2

    try:
        a, b = ev.pixel_array, de.pixel_array
    except Exception as exc:  # noqa: BLE001 - undecodable transfer syntax
        return [], f"pixels_undecodable:{type(exc).__name__}"
    if a.shape != b.shape:
        return [], "pixel_shape_differs"
    frames_a = a if a.ndim == 3 and getattr(ev, "NumberOfFrames", 1) and int(getattr(ev, "NumberOfFrames", 1)) > 1 else a[None]
    frames_b = b if frames_a is a else b[None]
    boxes = []
    for f, (fa, fb) in enumerate(zip(frames_a, frames_b)):
        diff = np.any(fa != fb, axis=-1) if fa.ndim == 3 else (fa != fb)
        if not diff.any():
            continue
        if masks is not None:
            masks.append((f, diff))
        mask = cv2.dilate(diff.astype(np.uint8), np.ones((5, 5), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for k in range(1, n):
            x, y, w, h, area = st[k]
            if area >= MIN_AREA:
                boxes.append({"frame": f, "bbox": [int(x), int(y), int(x + w), int(y + h)], "area": int(area)})
    return boxes, "ok"


def build(tcia: Path, out: Path, n: int = 48, seed: int = SEED) -> dict:
    import pydicom

    cw = {r["evaluation"]: r["de-identified"] for r in csv.DictReader((tcia / "Pseudo-PHI-DICOM-Dataset-uid_crosswalk.csv").read_text(encoding="utf-8").splitlines())}
    files = sorted((tcia / "pseudo_phi_dicom_data").rglob("*.dcm"))
    ev_idx, de_idx = {}, {}
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        (de_idx if str(ds.PatientID).startswith("Pseudo-PHI") else ev_idx)[ds.SOPInstanceUID] = (f, ds.Modality, str(ds.PatientID))
    pairs = [(u, ev_idx[u], de_idx[cw[u]]) for u in sorted(ev_idx) if cw.get(u) in de_idx]

    # 1) burned-in candidates: scan every pair's pixels (cheap; most are small PET slices)
    records = []
    for u, (ef, mod, pid), (dfp, _, _) in pairs:
        ev, de = pydicom.dcmread(ef), pydicom.dcmread(dfp)
        boxes, status = burned_boxes(ev, de)
        records.append({"uid": u, "eval": ef, "deid": dfp, "modality": mod, "patient": pid, "boxes": boxes,
                        "pixel_status": status, "frames": int(getattr(ev, "NumberOfFrames", 1) or 1)})
    rng = np.random.default_rng(seed)
    burned = [r for r in records if r["boxes"]]
    chosen = list(burned)
    rest = [r for r in records if not r["boxes"]]
    by_mod = defaultdict(list)
    for r in rest:
        by_mod[r["modality"]].append(r)
    for mod in by_mod:
        rng.shuffle(by_mod[mod])
        # prefer one instance per (patient, modality) first, for diversity
        seen, first, later = set(), [], []
        for r in by_mod[mod]:
            (later if r["patient"] in seen else first).append(r)
            seen.add(r["patient"])
        by_mod[mod] = first + later
    mods = sorted(by_mod)
    k = 0
    while len(chosen) < n and any(by_mod.values()):
        m = mods[k % len(mods)]
        if by_mod[m]:
            chosen.append(by_mod[m].pop(0))
        k += 1

    # 2) full GT for the selection
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    rows, gt_lines, stats = [], [], Counter()
    for i, r in enumerate(sorted(chosen, key=lambda r: (r["modality"], r["uid"])), start=1):
        did = f"DICOM_{i:03d}"
        dst = out / "inputs" / f"{did}.dcm"
        shutil.copyfile(r["eval"], dst)
        ev, de = pydicom.dcmread(r["eval"], stop_before_pixels=True), pydicom.dcmread(r["deid"], stop_before_pixels=True)
        hd = header_diff(ev, de)
        mask_rel = None
        if r["boxes"]:  # exact text-pixel masks (eval != curated) for pixel-level burned-in scoring
            masks: list = []
            burned_boxes(pydicom.dcmread(r["eval"]), pydicom.dcmread(r["deid"]), masks)
            (out / "masks").mkdir(exist_ok=True)
            mask_rel = f"masks/{did}.npz"
            np.savez_compressed(out / mask_rel, **{f"frame_{f}": m for f, m in masks})
        phi_tags = hd["removed"] + hd["changed"] + hd["emptied"]
        gt_lines.append(json.dumps({"dicom_id": did, "modality": r["modality"], "transfer_syntax": str(ev.file_meta.TransferSyntaxUID),
                                    "frames": r["frames"], "header": hd, "burned_in": r["boxes"], "burned_mask": mask_rel,
                                    "pixel_status": r["pixel_status"]}))
        rows.append({"dicom_id": did, "modality": r["modality"], "frames": r["frames"],
                     "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
                     "phi_tags": len(phi_tags), "uid_remapped": len(hd["uid_remapped"]),
                     "private_phi_tags": sum(1 for t in phi_tags if t["private"]),
                     "burned_in_boxes": len(r["boxes"]), "pixel_status": r["pixel_status"]})
        stats[f"modality={r['modality']}"] += 1
        for t in phi_tags:
            stats["tag:" + t["tag"].split("[")[0].split(".")[-1]] += 1
    (out / "gt.jsonl").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    summary = {"pairs": len(pairs), "pairs_with_burned_in": len(burned), "selected": len(rows),
               "pixel_status": dict(Counter(r["pixel_status"] for r in records)),
               "burned_by_modality": dict(Counter(r["modality"] for r in burned)), **{k: v for k, v in stats.items() if k.startswith("modality")}}
    return {"summary": summary, "rows": rows, "tag_counts": {k[4:]: v for k, v in stats.most_common() if k.startswith("tag:")}}


def write_committed(res: dict, csv_path: Path, md_path: Path) -> None:
    rows = res["rows"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    s = res["summary"]
    md = ["# DICOM ground truth", "",
          "Generated by `python -m data_prep.dicom_gt` from TCIA Pseudo-PHI-DICOM-Data (DOI 10.7937/s17z-r072, CC BY 4.0).",
          "Selection: `splits/dicom_selection.csv` (ids, modality, sha256, counts - no PHI values). Full GT incl. values",
          "stays in `data/dicom_gt/gt.jsonl` (gitignored).", "",
          "## How it was derived", "",
          "- Evaluation instances (synthetic PHI) are paired with their TCIA-curated de-identified twins through",
          "  `Pseudo-PHI-DICOM-Dataset-uid_crosswalk.csv` (SOPInstanceUID -> curated SOPInstanceUID).",
          "- Header PHI = every element (sequences recursed, PixelData / file meta excluded) the curator removed, emptied",
          "  or changed; remapped UIDs are counted separately (expected action: replace consistently).",
          f"- Burned-in text = pixels that differ between the pair (same shape), dilated 5x5, connected components >= {MIN_AREA} px.", "",
          "## Numbers", "",
          (f"- Paired instances: {s['pairs']}; pairs with burned-in pixel differences: {s['pairs_with_burned_in']} "
          f"({', '.join(f'{k} {v}' for k, v in s['burned_by_modality'].items()) or 'none'})."),
          f"- Pixel comparison status over all pairs: {', '.join(f'{k} {v}' for k, v in s['pixel_status'].items())}.",
          f"- Selected: {s['selected']} instances - " + ", ".join(f"{k.split('=')[1]} {v}" for k, v in s.items() if k.startswith('modality=')) + ".", "",
          "Most frequent PHI-bearing elements in the selection (count of instances):", "",
          "| Element | Instances |", "|---|---|",
          *(f"| {k} | {v} |" for k, v in list(res["tag_counts"].items())[:25]), "",
          "## Where this ground truth may be incomplete", "",
          "- It encodes TCIA's curation policy: elements TCIA kept (for example some dates or device fields) are not labelled",
          "  PHI here even if a stricter policy would remove them. Scores therefore measure agreement with TCIA, and our",
          "  de-identifier removing *more* than TCIA shows up as extra actions, not as misses.",
          "- Burned-in text that the curator did not clean, or cleaned by re-encoding / cropping (shape change), cannot be",
          "  recovered from a pixel diff; those instances carry `pixel_status` other than `ok` or no boxes.",
          "- Two evaluation instances have no curated twin on disk and are excluded.",
          "- Pixel-layout elements the curator rewrote when re-encoding (" + ", ".join(sorted(NOT_PHI)) + ") are",
          "  curation artefacts, not identity, and are excluded from header PHI.", ""]
    md_path.write_text("\n".join(md), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m data_prep.dicom_gt")
    ap.add_argument("--tcia", type=Path, default=Path("data/tcia"))
    ap.add_argument("--out", type=Path, default=Path("data/dicom_gt"))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    res = build(a.tcia, a.out, a.n, a.seed)
    write_committed(res, Path("splits/dicom_selection.csv"), Path("data_prep/DICOM_GT.md"))
    print(json.dumps(res["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
