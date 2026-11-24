"""Turn rendered clean pages into the messy corpus: quality tiers, file formats, ground truth.

    python -m data_prep.generate [--rendered data/generated] [--out data/corpus] [--seed 140]

Steps (reproducible, seed 140):

1. Assign each document ONE quality tier - ``clean_scan`` / ``poor_scan`` / ``phone_photo`` - by a
   seeded shuffle dealt round-robin, so the tiers are balanced before the dev/golden split
   stratifies on them (``data_prep.split``).
2. Degrade every page with ``data_prep.degrade.degrade_document`` (one composed homography per
   page; GT boxes pushed through it as quads and clipped axis-aligned boxes).
3. Export: multi-page documents alternate multi-page **PDF** / multi-page **TIFF**; single pages
   are **PNG** (clean, most poor scans), **JPEG** (phone photos) or a one-page PDF (every 4th clean
   scan / 3rd poor scan) so every format appears in every tier where it makes sense. PDFs embed the
   pages at 200 DPI, so the pipeline's 200-DPI raster reproduces the GT pixel grid exactly.

Outputs under ``<out>`` (gitignored): ``files/<doc_id>.<ext>`` (the inputs), ``gt/<doc_id>.json``
(ground truth in the FILE's page pixel coordinates + the homography back to the clean page),
``manifest.csv`` (ids, tier, format, flags, sha256) and ``overlays/`` (GT boxes drawn on sample
pages for visual QA).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from data_prep.degrade import degrade_document, write_page, write_pdf, write_tiff

SEED = 140
TIERS = ("clean_scan", "poor_scan", "phone_photo")


def assign_tiers(doc_ids: list[str], seed: int = SEED) -> dict[str, str]:
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(doc_ids))
    return {doc_ids[j]: TIERS[k % len(TIERS)] for k, j in enumerate(order)}


def choose_format(n_pages: int, tier: str, k: int) -> str:
    """k = running index of documents in this tier (deterministic format mix)."""
    if n_pages > 1:
        return "pdf" if k % 2 == 0 else "tiff"
    if tier == "phone_photo":
        return "jpg"
    if tier == "clean_scan":
        return "pdf" if k % 4 == 3 else "png"
    return "pdf" if k % 3 == 2 else "png"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(rendered: Path, out: Path, seed: int = SEED, n_overlays: int = 24) -> dict:
    gts = sorted((rendered / "gt_clean").glob("*.json"))
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in gts]
    tiers = assign_tiers([d["doc_id"] for d in docs], seed)
    doc_seeds = np.random.default_rng(seed + 2).integers(0, 2**31 - 1, size=len(docs))
    for sub in ("files", "gt", "overlays"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    per_tier_k: Counter = Counter()
    rows, stats = [], Counter()
    overlay_pick = set(np.random.default_rng(seed + 3).choice(len(docs), size=min(n_overlays, len(docs)),
                                                              replace=False).tolist())
    for i, d in enumerate(docs):
        tier = tiers[d["doc_id"]]
        pages = [np.asarray(Image.open(rendered / "clean" / p["image"]).convert("RGB")) for p in d["pages"]]
        boxes = [[tuple(inst["bbox"]) for inst in p["instances"]] for p in d["pages"]]
        results = degrade_document(pages, boxes, tier=tier, seed=int(doc_seeds[i]))
        fmt = choose_format(len(pages), tier, per_tier_k[tier])
        per_tier_k[tier] += 1
        imgs = [r.image for r in results]
        path = out / "files" / f"{d['doc_id']}.{fmt}"
        if fmt == "pdf":
            write_pdf(imgs, path, dpi=200)
        elif fmt == "tiff":
            write_tiff(imgs, path, dpi=200)
        else:
            write_page(imgs[0], path, jpeg_quality=85, dpi=200)
        gt_pages = []
        for p, r in zip(d["pages"], results):
            insts = []
            for inst, tb in zip(p["instances"], r.boxes):
                tbd = tb.to_dict()
                insts.append({**{k: inst[k] for k in ("field", "text", "handwritten", "phi")},
                              "bbox": tbd["aabb"], "quad": tbd["quad"], "bbox_clean": inst["bbox"]})
            rot = next((o.get("deg") for o in r.ops if o.get("op") == "rotate90"), 0)
            stats[f"rotate90={rot}"] += 1
            gt_pages.append({"page": p["page"], "size": list(r.size), "H": np.asarray(r.H).round(8).tolist(),
                             "clean_size": p["size"], "degrade": r.manifest_record(), "instances": insts})
        spec = d["spec"]
        gt = {"doc_id": d["doc_id"], "report_id": spec["report_id"], "file": path.name, "format": fmt, "tier": tier,
              "n_pages": len(pages), "seed": int(doc_seeds[i]), "multipage": spec["multipage"],
              "handwriting": spec["handwriting"], "hw_font": spec["hw_font"], "injection": spec["injection"],
              "template": spec["template"], "mesh_major": spec["mesh_major"], "expected": d["expected"],
              "pages": gt_pages}
        (out / "gt" / f"{d['doc_id']}.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
        rows.append({"doc_id": d["doc_id"], "report_id": spec["report_id"], "tier": tier, "format": fmt,
                     "n_pages": len(pages), "handwriting": int(spec["handwriting"]),
                     "injected": int(bool(spec["injection"])), "multipage": int(spec["multipage"]),
                     "sha256": _sha(path)})
        stats[f"tier={tier}"] += 1
        stats[f"format={fmt}"] += 1
        if i in overlay_pick:
            _overlay(imgs[0], gt_pages[0]["instances"], out / "overlays" / f"{d['doc_id']}_{tier}.jpg")
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"docs": len(rows), "pages": sum(r["n_pages"] for r in rows), **dict(sorted(stats.items()))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def _overlay(img: np.ndarray, insts: list[dict], path: Path) -> None:
    im = Image.fromarray(img).convert("RGB")
    dr = ImageDraw.Draw(im)
    for it in insts:
        color = (220, 0, 0) if it["phi"] else ((0, 150, 0) if it["handwritten"] else (0, 80, 255))
        dr.polygon([tuple(pt) for pt in it["quad"]], outline=color, width=3)
    im.thumbnail((1000, 1000))
    im.save(path, quality=85)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m data_prep.generate")
    ap.add_argument("--rendered", type=Path, default=Path("data/generated"))
    ap.add_argument("--out", type=Path, default=Path("data/corpus"))
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    print(json.dumps(build(a.rendered, a.out, a.seed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
