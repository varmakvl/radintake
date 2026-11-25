"""Score PDF text-layer use and table reading on the synthetic table set (data_prep.tables_set).

Conditions, each a full pipeline run (``pipeline.process_file``, OCR-first):

* ``digital``          - digital PDFs, text layer on (the new default);
* ``digital_ocr``      - the same PDFs with ``loader.pdf_text_layer.enabled: false`` (old behaviour);
* ``scan_clean`` / ``scan_poor`` - the rasterised, degraded, image-only PDFs (OCR + per-cell re-read).

Metrics per condition (aggregates only; the set is synthetic, values never leave ``runs/``):

* tables: detection recall (a GT table is found when a predicted table on the page matches its
  shape-independent cell text best, with cell accuracy > 0), exact shape rate, cell exact match
  (normalised text, GT cells counted; missing rows / columns count wrong) and cell CER;
* fields: exact match and CER over the report fields in the GT (same normalisation as eval.metrics);
* findings_clean: findings exact match - table text must not leak into the prose;
* hidden text -> ``prompt_injection_suspected`` recall / false-positive rate (digital conditions);
* status mix and seconds per document.

    python -m radintake.eval.tables_eval [--set data/tables_set] [--out runs/tables_eval] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from radintake.eval.metrics import cer, norm

INJ = "prompt_injection_suspected"
CONDITIONS = {"digital": ("digital", True), "digital_ocr": ("digital", False),
              "scan_clean": ("scan_clean", True), "scan_poor": ("scan_poor", True)}


def _cell_scores(gt_rows: list[list[str]], pred: dict[str, Any] | None) -> tuple[int, int, list[float], bool]:
    """(correct cells, GT cells, per-cell CER, exact shape) of one GT table vs one predicted table."""
    n = sum(len(r) for r in gt_rows)
    if pred is None:
        return 0, n, [1.0] * n, False
    grid = {(c["row"], c["col"]): c["text"] for c in pred["cells"]}
    ok, cers = 0, []
    for r, row in enumerate(gt_rows):
        for c, g in enumerate(row):
            p = grid.get((r, c), "")
            ok += norm(p) == norm(g)
            cers.append(cer(p, g))
    shape = pred["n_rows"] == len(gt_rows) and pred["n_cols"] == max(len(r) for r in gt_rows)
    return ok, n, cers, shape


def match_tables(gt_tables: list[dict], pred_tables: list[dict]) -> list[dict | None]:
    """Greedy one-to-one match of GT tables to predicted tables by correct cells."""
    pairs = []
    for gi, g in enumerate(gt_tables):
        for pi, p in enumerate(pred_tables):
            ok = _cell_scores(g["rows"], p)[0]
            if ok:
                pairs.append((ok, gi, pi))
    out: list[dict | None] = [None] * len(gt_tables)
    used: set[int] = set()
    for _ok, gi, pi in sorted(pairs, reverse=True):
        if out[gi] is None and pi not in used:
            out[gi] = pred_tables[pi]
            used.add(pi)
    return out


def score_doc(pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    tables = pred.get("tables") or []
    matched = match_tables(gt["tables"], tables)
    t_ok = t_n = 0
    cers: list[float] = []
    shapes = found = 0
    per_style: dict[str, list[int]] = {}
    for g, m in zip(gt["tables"], matched, strict=True):
        ok, n, c, shape = _cell_scores(g["rows"], m)
        t_ok, t_n = t_ok + ok, t_n + n
        cers += c
        shapes += shape
        found += m is not None
        s = per_style.setdefault(g["style"], [0, 0])
        s[0], s[1] = s[0] + ok, s[1] + n
    rep = pred.get("report") or {}
    fields = {}
    for k, v in gt["fields"].items():
        pv = (rep.get(k) or {}).get("value") or ""
        fields[k] = {"em": norm(pv) == norm(v), "cer": cer(pv, v)}
    return {"status": pred.get("status"), "flags": pred.get("flags") or [], "hidden_gt": gt["hidden_text"],
            "tables_gt": len(gt["tables"]), "tables_found": found, "tables_pred": len(tables), "shapes_ok": shapes,
            "cells_ok": t_ok, "cells_n": t_n, "cell_cer": cers, "per_style": per_style, "fields": fields}


def aggregate(recs: list[dict[str, Any]], secs: list[float]) -> dict[str, Any]:
    def rate(a: float, b: float) -> float | None:
        return round(a / b, 4) if b else None

    tg = sum(r["tables_gt"] for r in recs)
    styles: dict[str, list[int]] = {}
    for r in recs:
        for s, (ok, n) in r["per_style"].items():
            styles.setdefault(s, [0, 0])
            styles[s][0] += ok
            styles[s][1] += n
    fnames = sorted({k for r in recs for k in r["fields"]})
    hid = [r for r in recs if r["hidden_gt"]]
    clean = [r for r in recs if not r["hidden_gt"]]
    statuses: dict[str, int] = {}
    for r in recs:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {
        "docs": len(recs),
        "tables": {"gt": tg, "detected": rate(sum(r["tables_found"] for r in recs), tg),
                   "shape_exact": rate(sum(r["shapes_ok"] for r in recs), tg),
                   "cell_exact_match": rate(sum(r["cells_ok"] for r in recs), sum(r["cells_n"] for r in recs)),
                   "cell_cer": round(statistics.fmean(c for r in recs for c in r["cell_cer"]), 4) if recs else None,
                   "extra_predicted": sum(max(0, r["tables_pred"] - r["tables_found"]) for r in recs),
                   "cell_em_by_style": {s: rate(ok, n) for s, (ok, n) in sorted(styles.items())}},
        "fields": {"mean_em": rate(sum(r["fields"][f]["em"] for r in recs for f in fnames), len(recs) * len(fnames)),
                   "mean_cer": round(statistics.fmean(r["fields"][f]["cer"] for r in recs for f in fnames), 4),
                   "per_field_em": {f: rate(sum(r["fields"][f]["em"] for r in recs), len(recs)) for f in fnames}},
        # hidden text is reported under the v1 flag prompt_injection_suspected (+ review reason hidden_text)
        "hidden_text": {"docs_with_hidden": len(hid),
                        "flag_recall": rate(sum(INJ in r["flags"] for r in hid), len(hid)),
                        "flag_false_positive": rate(sum(INJ in r["flags"] for r in clean), len(clean))},
        "status": statuses,
        "seconds_per_doc": {"p50": round(statistics.median(secs), 2), "max": round(max(secs), 2)} if secs else None,
    }


def run(set_dir: Path, out: Path, limit: int | None = None, conditions: list[str] | None = None) -> dict[str, Any]:
    from radintake.config import load_config
    from radintake.pipeline import process_file

    gts = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((set_dir / "gt").glob("*.json"))}
    ids = sorted(gts)[:limit] if limit else sorted(gts)
    with tempfile.TemporaryDirectory() as td:
        off = Path(td) / "no_text_layer.yaml"
        off.write_text("loader:\n  pdf_text_layer:\n    enabled: false\n", encoding="utf-8")
        cfg_on, cfg_off = load_config(), load_config(off)
        result: dict[str, Any] = {"set": str(set_dir), "conditions": {}}
        for name in conditions or list(CONDITIONS):
            variant, text_layer = CONDITIONS[name]
            recs, secs = [], []
            for doc_id in ids:
                f = set_dir / "files" / f"{doc_id}_{variant}.pdf"
                t0 = time.perf_counter()
                pred = process_file(f, out / name, config=cfg_on if text_layer else cfg_off)
                secs.append(time.perf_counter() - t0)
                recs.append(score_doc(pred, gts[doc_id]))
            result["conditions"][name] = aggregate(recs, secs)
            print(json.dumps({name: result["conditions"][name]["tables"]}), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default="data/tables_set")
    ap.add_argument("--out", default="runs/tables_eval")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--conditions", nargs="*", choices=list(CONDITIONS))
    ap.add_argument("--results", default="eval/results/tables_eval.json",
                    help="aggregate metrics (no values) written here")
    a = ap.parse_args(argv)
    res = run(Path(a.set), Path(a.out), a.limit, a.conditions)
    Path(a.results).parent.mkdir(parents=True, exist_ok=True)
    Path(a.results).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
