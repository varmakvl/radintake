"""Step inspector: dump the input and output of every pipeline step for one or more files.

    python -m radintake.inspect <file-or-folder> [--out runs/inspect] [--pipeline ocr_first|vlm_first|hybrid]

For each input it runs the normal pipeline with ``RADINTAKE_INSPECT_DIR`` set; the planner then hands
every successful stage to :func:`dump_stage`, which writes, under ``<out>/<input name>/``::

    00_input/        page_N.png (as loaded: original pixel grid), pages.json
    10_load/         stage output summary
    20_preprocess/   page_N.png (processed), decisions.json, transform.json
    30_phi/          page_N_ocr.png (OCR word boxes), page_N_masked.png (what any model may see),
                     page_N_phi.png (detections on the processed page), ocr_words_N.json, detections.json
    30_dicom_deid/   before.png / after.png (frame 0, display-normalised), header_changes.json
    40_extract/      report.json (values, confidence, provenance), page_N_fields.png (boxes on the
                     ORIGINAL page), vlm_calls.json + vlm_response_K.json when a VLM was called
    50_verify/       fields.json (grounded / needs_review / low_confidence), flags.json
    55_injection/    flags.json, untrusted text
    99_output/       the final cli_output.v1 JSON (+ de-identified .dcm for DICOM)
    index.html       all steps side by side - open in a browser

LOCAL DEBUGGING ONLY: these files contain PHI and report text (the point is to see them). They are
written under ``runs/`` which is gitignored; never commit or share them.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

ENV = "RADINTAKE_INSPECT_DIR"
_ORDER = {"load": "10", "preprocess": "20", "tables": "25", "phi": "30", "dicom_deid": "30", "extract": "40",
          "verify": "50", "injection": "55"}


def enabled() -> Path | None:
    v = os.environ.get(ENV)
    return Path(v) if v else None


def _case_dir(ctx: Any) -> Path:
    root = enabled()
    d = root / ctx.input_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _png(img: Any, path: Path, max_side: int = 0) -> None:
    from PIL import Image

    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = a.astype(np.float64)
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        a = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    im = Image.fromarray(a)
    if max_side:
        im.thumbnail((max_side, max_side))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def _boxes_png(img: Any, boxes: list[tuple[list[float], tuple[int, int, int], str]], path: Path) -> None:
    from PIL import Image, ImageDraw

    im = Image.fromarray(np.asarray(img)).convert("RGB")
    d = ImageDraw.Draw(im)
    for b, color, label in boxes:
        d.rectangle([float(v) for v in b], outline=color, width=3)
        if label:
            d.text((float(b[0]), max(0.0, float(b[1]) - 12)), label[:28], fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def _box(b: Any) -> list[float]:
    if hasattr(b, "x0"):
        return [b.x0, b.y0, b.x1, b.y1]
    return list(b)[:4]


def _jsonable(o: Any, depth: int = 0) -> Any:
    if depth > 6:
        return str(o)[:200]
    if isinstance(o, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple, set)):
        o = list(o)
        return [_jsonable(v, depth + 1) for v in o[:500]]
    if isinstance(o, np.ndarray):
        return f"<array {o.shape} {o.dtype}>"
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if hasattr(o, "x0") and hasattr(o, "y1"):
        return _box(o)
    if hasattr(o, "__dict__"):
        return {"type": type(o).__name__, **{k: _jsonable(v, depth + 1) for k, v in vars(o).items()
                                             if not k.startswith("_") and k != "image"}}
    return str(o)[:200]


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=1, ensure_ascii=False), encoding="utf-8")


def _trace_events(ctx: Any, kind: str) -> list[dict]:
    p = getattr(ctx.trace, "path", None)
    if not p or not Path(p).is_file():
        return []
    out = []
    for ln in Path(p).read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if e.get("kind") == kind:
            out.append(e.get("data") or {})
    return out


# ------------------------------------------------------------------------------------- per stage
def dump_stage(name: str, ctx: Any, updates: dict[str, Any] | None, duration_ms: float) -> None:
    """Called by the planner after each successful stage when RADINTAKE_INSPECT_DIR is set."""
    case = _case_dir(ctx)
    d = case / f"{_ORDER.get(name, '60')}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    u = updates or {}
    meta = {"stage": name, "duration_ms": duration_ms, "input_type": ctx.input_type, "keys": sorted(u)}
    fn = globals().get(f"_dump_{name}")
    if fn:
        fn(ctx, u, d)
    else:
        _write(d / "updates.json", u)
    _write(d / "stage.json", meta)


def _dump_load(ctx, u, d):
    pages = u.get("pages") or []
    inp = d.parent / "00_input"
    for pg in pages:
        _png(pg.image, inp / f"page_{pg.index}.png")
    _write(inp / "pages.json", [{"index": p.index, "size": list(np.asarray(p.image).shape), "source": p.source,
                                 "meta": p.meta} for p in pages])
    _write(d / "summary.json", {"n_pages": u.get("n_pages")})


def _dump_preprocess(ctx, u, d):
    for pg in u.get("pages") or []:
        _png(pg.image, d / f"page_{pg.index}.png")
    _write(d / "transform.json", {p.index: np.asarray(p.transform.matrix).round(6).tolist() for p in u.get("pages") or []})
    _write(d / "decisions.json", _trace_events(ctx, "preprocess_decision"))


def _dump_tables(ctx, u, d):
    """Detected tables drawn on the processed page (red cells) + the tables.v1 JSON."""
    import cv2

    pages = {p.index: p for p in (ctx.data.get("pages") or [])}
    for pidx, tables in (u.get("table_objs") or {}).items():
        pg = pages.get(pidx)
        if pg is None:
            continue
        img = np.asarray(pg.image).copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for t in tables:
            for c in t.cells:
                b = c.bbox
                cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), (220, 0, 0), 2)
        _png(img, d / f"page_{pidx}_tables.png")
    _write(d / "tables.json", u.get("tables") or [])
    _write(d / "text_layer.json", {p.index: p.meta.get("text_layer") for p in pages.values()})


def _dump_phi(ctx, u, d):
    pages = {p.index: p for p in (u.get("pages") or ctx.data.get("pages") or [])}
    words = u.get("ocr_words") or {}
    dets = (u.get("phi") or {}).get("detections") or []
    for idx, pg in pages.items():
        ws = words.get(idx) or []
        _write(d / f"ocr_words_{idx}.json", [{"text": w.text, "bbox": _box(w.bbox),
                                              "conf": round(float(getattr(w, "confidence", getattr(w, "conf", 1.0))), 3)}
                                             for w in ws])
        _boxes_png(pg.image, [(_box(w.bbox), (0, 120, 255), "") for w in ws], d / f"page_{idx}_ocr.png")
        # detections are reported in ORIGINAL coordinates: draw them on the loaded page
        orig = (ctx.data.get("original_pages") or [None] * idx)
        base = next((p.image for p in orig if p is not None and p.index == idx), pg.image)
        _boxes_png(base, [(x["bbox"], (220, 0, 0), x.get("type", "")) for x in dets if x.get("page") == idx],
                   d / f"page_{idx}_phi_on_original.png")
    for k, pl in enumerate(u.get("phi_masked_pages") or [], start=1):
        data = pl.data if hasattr(pl, "data") else pl.get("data")
        _png(data["image"], d / f"page_{k}_masked.png")
    _write(d / "detections.json", dets)
    _write(d / "summary.json", {"masked_artifacts": (u.get("phi") or {}).get("masked_artifacts"),
                                "flags": u.get("flags"), "n_detections": len(dets),
                                "note": "page_N_masked.png is exactly what a VLM may receive"})


def _dump_dicom_deid(ctx, u, d):
    import pydicom

    info = u.get("dicom") or {}
    _write(d / "summary.json", info)
    try:
        src = pydicom.dcmread(str(ctx.path))
        _png(src.pixel_array[0] if src.pixel_array.ndim == 3 and int(getattr(src, "NumberOfFrames", 1) or 1) > 1
             else src.pixel_array, d / "before.png")
        if info.get("output_path"):
            out = pydicom.dcmread(str(ctx.out_dir / info["output_path"]))
            _png(out.pixel_array[0] if out.pixel_array.ndim == 3 and int(getattr(out, "NumberOfFrames", 1) or 1) > 1
                 else out.pixel_array, d / "after.png")
            changes = []
            out_vals = {el.keyword or str(el.tag): str(el.value)[:120] for el in out if el.tag != 0x7FE00010}
            for el in src:
                if el.tag == 0x7FE00010 or el.VR == "SQ":
                    continue
                k = el.keyword or str(el.tag)
                before, after = str(el.value)[:120], out_vals.get(k, "<removed>")
                if before != after:
                    changes.append({"tag": k, "before": before, "after": after})
            _write(d / "header_changes.json", changes)
    except Exception as exc:  # noqa: BLE001 - inspection never fails the case
        _write(d / "error.json", {"error": f"{type(exc).__name__}: {exc}"[:300]})


def _dump_extract(ctx, u, d):
    rep = u.get("report") or {}
    _write(d / "report.json", rep)
    _write(d / "extraction.json", u.get("extraction"))
    originals = {p.index: p for p in (ctx.data.get("original_pages") or ctx.data.get("pages") or [])}
    colors = {"ocr": (0, 120, 255), "vlm": (200, 0, 200), "hybrid": (0, 160, 0)}
    per_page: dict[int, list] = {}
    for f, fld in rep.items():
        for p in (fld or {}).get("provenance") or []:
            per_page.setdefault(int(p["page"]), []).append((p["bbox"], colors.get(p.get("source"), (0, 0, 0)), f))
    for idx, boxes in per_page.items():
        if idx in originals:
            _boxes_png(originals[idx].image, boxes, d / f"page_{idx}_fields.png")
    calls = _trace_events(ctx, "model_call")
    _write(d / "vlm_calls.json", calls)
    from radintake.tools.vlm_client import vlm_params

    p = vlm_params(ctx.config)
    cache = Path(p.get("cache_dir") or (Path(ctx.config.data_dir) / "cache" / "vlm"))
    for k, c in enumerate(calls, start=1):
        key = hashlib.sha256(f"{c.get('model_id')}|{c.get('payload_hash')}|{c.get('prompt_hash')}".encode()).hexdigest()
        if (cache / f"{key}.json").is_file():
            shutil.copyfile(cache / f"{key}.json", d / f"vlm_response_{k}.json")


def _dump_verify(ctx, u, d):
    rep = u.get("report") or ctx.data.get("report") or {}
    _write(d / "fields.json", {f: {k: (v or {}).get(k) for k in ("grounded", "needs_review", "low_confidence", "confidence")}
                               for f, v in rep.items()})
    _write(d / "flags.json", u.get("flags"))
    _write(d / "updates.json", {k: v for k, v in u.items() if k != "report"})


def _dump_injection(ctx, u, d):
    _write(d / "flags.json", u.get("flags"))
    _write(d / "updates.json", {k: v for k, v in u.items() if k != "report"})


# ------------------------------------------------------------------------------------- index page
def write_index(case: Path, output_json: Path | None) -> Path:
    out = case / "99_output"
    out.mkdir(exist_ok=True)
    if output_json and output_json.is_file() and output_json.resolve() != (out / output_json.name).resolve():
        shutil.copyfile(output_json, out / output_json.name)
    parts = [f"<!doctype html><meta charset='utf-8'><title>{html.escape(case.name)} - steps</title>",
             ("<style>body{font:14px system-ui;margin:16px;background:#fafafa}h2{border-top:2px solid #ccc;padding-top:8px}"
             ".row{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;background:#fff;padding:6px;border:1px solid #ddd}"
             "img{max-width:520px;max-height:680px;display:block}pre{background:#fff;border:1px solid #ddd;padding:8px;"
             "max-height:360px;overflow:auto;white-space:pre-wrap;max-width:100%}</style>"),
             f"<h1>{html.escape(case.name)}</h1><p>Local debug view - contains PHI. Steps in pipeline order.</p>"]
    for step in sorted(p for p in case.iterdir() if p.is_dir()):
        parts.append(f"<h2>{html.escape(step.name)}</h2><div class='row'>")
        for img in sorted(step.glob("*.png")):
            parts.append(f"<figure><img src='{step.name}/{img.name}'><figcaption>{img.name}</figcaption></figure>")
        parts.append("</div>")
        for js in sorted(step.glob("*.json")):
            txt = js.read_text(encoding="utf-8")
            parts.append(f"<details{' open' if js.name in ('report.json', 'decisions.json', 'header_changes.json') else ''}>"
                         f"<summary>{js.name}</summary><pre>{html.escape(txt[:20000])}</pre></details>")
    idx = case / "index.html"
    idx.write_text("\n".join(parts), encoding="utf-8")
    return idx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m radintake.inspect", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="a file or a folder of files")
    ap.add_argument("--out", type=Path, default=Path("runs/inspect"))
    ap.add_argument("--pipeline", choices=("ocr_first", "vlm_first", "hybrid"))
    a = ap.parse_args(argv)
    os.environ[ENV] = str(a.out.resolve())
    if a.pipeline:
        os.environ["RADINTAKE_EXTRACTION_PIPELINE"] = a.pipeline
    from radintake.config import load_config
    from radintake.pipeline import process_file
    from radintake.tracing.writer import TraceWriter

    cfg = load_config()
    files = sorted(p for p in a.path.iterdir() if p.is_file()) if a.path.is_dir() else [a.path]
    tw = TraceWriter(a.out / "_traces", model_ids=cfg.model_ids, offline=cfg.offline)
    for f in files:
        case = a.out / f.name
        if case.exists():
            shutil.rmtree(case)
        doc = process_file(f, case / "99_output", config=cfg, trace_writer=tw)
        idx = write_index(case, case / "99_output" / f"{f.name}.json")
        print(f"{f.name}: {doc['status']} -> {idx}")
    tw.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
