"""Error analysis: the N worst pages of a run, every failure categorised, overlays drawn.

    python -m radintake.eval.error_analysis --run runs/<run> [--n 10] [--corpus data/corpus]

Ranking. A page's badness = mean over the report fields whose ground truth is on that page of
(CER, with an omission counted as 1.0), plus 0.25 per ground-truth PHI instance on the page that was not
detected. Ties break by doc id.

Categories (one per failed field, first rule that applies):

* ``omission``          - ground truth present, prediction empty;
* ``hallucination``     - prediction not in the page's printed text;
* ``handwriting``       - the field was handwritten on the page;
* ``wrong field``       - the prediction matches ANOTHER field's ground truth better than its own;
* ``layout: bleed``     - the prediction contains the ground truth plus extra text (reading order /
                          section boundary);
* ``layout: truncated`` - the prediction is a clear prefix / part of the ground truth;
* ``OCR misread``       - otherwise (characters wrong in the right place);
* ``PHI miss``          - per undetected PHI instance (type recorded).

Outputs in ``<run>/error_analysis/``: ``summary.json`` and ``summary.md`` (ids, fields, categories,
CER - no values, safe to commit), and ``<doc>_p<n>.png`` overlays (ground truth green, prediction
blue / purple for VLM, missed PHI red) - local only: they show licensed report pages and synthetic PHI.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from radintake.eval.metrics import FIELDS, cer, norm

REPORT_FIELDS = ("patient_id", "accession_number", "indication", "comparison", "findings", "impression")


def categorise(field: str, pred: str, gt: str, gt_all: dict[str, str], printed: str, handwritten: bool) -> str | None:
    p, g = norm(pred), norm(gt)
    if not g:
        return None
    if p == g:
        return None
    if not p:
        return "omission"
    if fuzz.partial_ratio(p, printed) < 80:
        return "hallucination"
    if handwritten:
        return "handwriting"
    own = fuzz.ratio(p, g)
    for f, other in gt_all.items():
        o = norm(other)
        if f != field and o and fuzz.ratio(p, o) > max(own, 80):
            return "wrong field"
    if len(p) > 1.15 * len(g) and fuzz.partial_ratio(g, p) >= 90:
        return "layout: bleed"
    if len(p) < 0.85 * len(g) and fuzz.partial_ratio(p, g) >= 90:
        return "layout: truncated"
    return "OCR misread"


def analyse(run: Path, corpus: Path, n: int = 10, draw: bool = True) -> dict[str, Any]:
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    scores = [json.loads(ln) for ln in (run / "scores.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    phi_miss = defaultdict(list)
    for r in scores:
        if r["kind"] == "phi_gt" and not r["detected"]:
            phi_miss[(r["doc_id"], r["page"])].append(r["type"])
    pages: list[dict] = []
    for doc in meta["doc_ids"]:
        gt = json.loads((corpus / "gt" / f"{doc}.json").read_text(encoding="utf-8"))
        out = run / "outputs" / f"{gt['file']}.json"
        if not out.is_file():
            continue
        pred = json.loads(out.read_text(encoding="utf-8"))
        rep = pred.get("report") or {}
        printed = " ".join(norm(i["text"]) for p in gt["pages"] for i in p["instances"])
        gt_all = {f: gt["expected"].get(f) or "" for f in FIELDS}
        for p in gt["pages"]:
            on_page = {i["field"]: i for i in p["instances"]}
            fails, cers = [], []
            for f in REPORT_FIELDS:
                if f not in on_page or not gt["expected"].get(f):
                    continue
                pv = (rep.get(f) or {}).get("value") or ""
                c = cer(norm(pv), norm(gt["expected"][f]))
                cers.append(c)
                cat = categorise(f, pv, gt["expected"][f], gt_all, printed, bool(on_page[f].get("handwritten")))
                if cat:
                    fails.append({"field": f, "category": cat, "cer": round(c, 3)})
            misses = phi_miss.get((doc, p["page"]), [])
            fails += [{"field": t, "category": "PHI miss", "cer": None} for t in misses]
            if not cers and not misses:
                continue
            bad = (sum(cers) / len(cers) if cers else 0.0) + 0.25 * len(misses)
            pages.append({"doc_id": doc, "page": p["page"], "tier": gt["tier"], "format": gt["format"],
                          "handwriting": bool(gt.get("handwriting")), "injected": bool(gt.get("injection")),
                          "badness": round(bad, 4), "mean_cer": round(sum(cers) / len(cers), 4) if cers else None,
                          "failures": fails, "_gt": gt, "_pred": pred})
    pages.sort(key=lambda x: (-x["badness"], x["doc_id"], x["page"]))
    worst = pages[:n]
    cats = Counter(f["category"] for w in worst for f in w["failures"])
    all_cats = Counter(f["category"] for w in pages for f in w["failures"])
    outdir = run / "error_analysis"
    outdir.mkdir(exist_ok=True)
    if draw:
        for w in worst:
            _overlay(w, corpus, outdir)
    clean = [{k: v for k, v in w.items() if not k.startswith("_")} for w in worst]
    summary = {"run": meta.get("label"), "split": meta.get("split"), "pipeline": meta.get("pipeline"),
               "n_pages_scored": len(pages), "worst": clean, "categories_worst": dict(cats.most_common()),
               "categories_all_pages": dict(all_cats.most_common()),
               "worst_by_tier": dict(Counter(w["tier"] for w in worst))}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (outdir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _markdown(s: dict) -> str:
    lines = [f"# Error analysis: {len(s['worst'])} worst pages ({s['split']}, {s['pipeline']})", "",
             f"Pages scored: {s['n_pages_scored']}. Worst pages by tier: "
             + ", ".join(f"{k} {v}" for k, v in s["worst_by_tier"].items()) + ".", "",
             "| # | Doc | Page | Tier | Format | Handwriting | Mean CER | Failures |", "|---|---|---|---|---|---|---|---|"]
    for k, w in enumerate(s["worst"], start=1):
        fails = "; ".join(f"{f['field']}: {f['category']}" for f in w["failures"])
        lines.append(f"| {k} | {w['doc_id']} | {w['page']} | {w['tier']} | {w['format']} | "
                     f"{'yes' if w['handwriting'] else 'no'} | {w['mean_cer'] if w['mean_cer'] is not None else '-'} | {fails} |")
    lines += ["", "## Failure categories", "", "| Category | Worst 10 pages | All scored pages |", "|---|---|---|"]
    for c in sorted(set(s["categories_all_pages"]) | set(s["categories_worst"]),
                    key=lambda c: -s["categories_all_pages"].get(c, 0)):
        lines.append(f"| {c} | {s['categories_worst'].get(c, 0)} | {s['categories_all_pages'].get(c, 0)} |")
    return "\n".join(lines) + "\n"


def _overlay(w: dict, corpus: Path, outdir: Path) -> None:
    from PIL import ImageDraw

    from radintake.orchestrator.router import route
    from radintake.tools.loader import load_pages

    path = corpus / "files" / w["_gt"]["file"]
    pages = load_pages(path, route(path).input_type)
    page = next((p for p in pages if p.index == w["page"]), None)
    if page is None:
        return
    from PIL import Image

    im = Image.fromarray(page.image).convert("RGB")
    d = ImageDraw.Draw(im)
    gp = next(p for p in w["_gt"]["pages"] if p["page"] == w["page"])
    for i in gp["instances"]:
        if i["field"] in FIELDS or i.get("phi"):
            d.rectangle(i["bbox"], outline=(0, 170, 0), width=3)
    missed = {f["field"] for f in w["failures"] if f["category"] == "PHI miss"}
    for i in gp["instances"]:
        if i["field"] in missed:
            d.rectangle([v - 4 if k < 2 else v + 4 for k, v in enumerate(i["bbox"])], outline=(220, 0, 0), width=4)
    for f, fld in ((w["_pred"].get("report") or {}).items()):
        for pv in fld.get("provenance") or []:
            if int(pv["page"]) == w["page"]:
                color = (170, 0, 200) if pv.get("source") == "vlm" else (0, 90, 255)
                d.rectangle(pv["bbox"], outline=color, width=2)
                d.text((pv["bbox"][0], max(0, pv["bbox"][1] - 12)), f, fill=color)
    im.thumbnail((1400, 1400))
    im.save(outdir / f"{w['doc_id']}_p{w['page']}.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m radintake.eval.error_analysis")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--no-overlays", action="store_true")
    a = ap.parse_args(argv)
    s = analyse(a.run, a.corpus, a.n, draw=not a.no_overlays)
    print(_markdown(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
