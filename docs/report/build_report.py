"""Build the assignment report (HTML + PDF) from a finished evaluation run.

    python docs/report/build_report.py --run runs/golden_final [--out docs/report/out]

Inputs: <run>/metrics.json, run.json, error_analysis/summary.json (run error_analysis first),
eval/results/pipeline_comparison.json, eval/results/attempts.md, the corpus (for overlays) and
data/dicom_gt (for the DICOM before/after). Output: <out>/report.html, <out>/report.pdf, <out>/img/*.

The output shows rendered IU report pages (CC BY-NC-ND) and synthetic PHI, so <out> is gitignored:
the report is shared with the assessor, never committed. This script is committed.
"""

from __future__ import annotations

import argparse
import html
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIERS = ("clean_scan", "poor_scan", "phone_photo")
TIER_LABEL = {"clean_scan": "Clean scan", "poor_scan": "Poor scan", "phone_photo": "Phone photo"}
FIELD_ORDER = ("patient_name", "patient_id", "date_of_birth", "accession_number", "exam_date",
               "indication", "comparison", "findings", "impression")


def pct(x, d=1):
    return "-" if x is None else f"{100 * x:.{d}f}%"


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def table(head: list[str], rows: list[list], cls: str = "") -> str:
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


# ---------------------------------------------------------------- images

def overlays(run: Path, corpus: Path, img: Path) -> list[tuple[str, str, dict]]:
    """Best / median / worst page by badness, drawn with the error-analysis overlay."""
    from radintake.eval import error_analysis as ea

    s = ea.analyse(run, corpus, n=100000, draw=False)
    pages = s["worst"]  # all scored pages, worst first
    picks = [("Worst", pages[0]), ("Median", pages[len(pages) // 2]), ("Best", pages[-1])]
    out = []
    for label, p in picks:
        gt = json.loads((corpus / "gt" / f"{p['doc_id']}.json").read_text(encoding="utf-8"))
        pred = json.loads((run / "outputs" / f"{gt['file']}.json").read_text(encoding="utf-8"))
        w = dict(p, _gt=gt, _pred=pred)
        ea._overlay(w, corpus, img)
        src = img / f"{p['doc_id']}_p{p['page']}.png"
        dst = img / f"overlay_{label.lower()}.png"
        src.replace(dst)
        out.append((label, dst.name, p))
    ea.analyse(run, corpus, n=10, draw=False)  # restore the 10-worst summary.json that the full ranking overwrote
    return out


def dicom_before_after(run: Path, img: Path, dicom_id: str) -> str:
    """Original vs de-identified: whole image, plus the burned-in text region zoomed (GT box + margin)."""
    import numpy as np
    import pydicom
    from PIL import Image, ImageDraw

    def to8(ds):
        a = ds.pixel_array.astype("float32")
        if a.ndim == 3 and a.shape[-1] != 3:
            a = a[0]
        lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
        a = np.clip((a - lo) / max(hi - lo, 1), 0, 1) * 255
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            a = 255 - a
        return Image.fromarray(a.astype("uint8")).convert("RGB")

    gt = next(json.loads(ln) for ln in (ROOT / "data/dicom_gt/gt.jsonl").read_text(encoding="utf-8").splitlines()
              if ln.strip() and json.loads(ln)["dicom_id"] == dicom_id)
    boxes = [b["bbox"] for b in (gt.get("burned_in") or []) if b["bbox"][2] - b["bbox"][0] > 20]
    before = to8(pydicom.dcmread(ROOT / "data" / "dicom_gt" / "inputs" / f"{dicom_id}.dcm"))
    after = to8(pydicom.dcmread(run / "outputs" / f"{dicom_id}.dcm.deid.dcm"))
    x0 = max(0, min(b[0] for b in boxes) - 40); y0 = max(0, min(b[1] for b in boxes) - 40)
    x1 = min(before.width, max(b[2] for b in boxes) + 40); y1 = min(before.height, max(b[3] for b in boxes) + 40)
    zb, za = before.crop((x0, y0, x1, y1)), after.crop((x0, y0, x1, y1))
    zw = 900
    zb = zb.resize((zw, max(1, int(zb.height * zw / zb.width)))); za = za.resize(zb.size)
    h = 420
    fb = before.resize((int(before.width * h / before.height), h)); fa = after.resize(fb.size)
    for im in (fb, fa):
        s = h / before.height
        ImageDraw.Draw(im).rectangle([x0 * s, y0 * s, x1 * s, y1 * s], outline=(220, 0, 0), width=2)
    W = max(fb.width * 2 + 20, zw)
    canvas = Image.new("RGB", (W, h + 20 + zb.height * 2 + 10), "white")
    canvas.paste(fb, (0, 0)); canvas.paste(fa, (fb.width + 20, 0))
    canvas.paste(zb, (0, h + 20)); canvas.paste(za, (0, h + 30 + zb.height))
    name = f"dicom_{dicom_id}.png"
    canvas.save(img / name)
    return name


def screenshot(url_or_file: Path, dst: Path, width=1200, height=900, full=False) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": width, "height": height})
        pg.goto(url_or_file.resolve().as_uri())
        pg.wait_for_timeout(300)
        pg.screenshot(path=str(dst), full_page=full)
        b.close()


def trace_excerpt(run: Path) -> str:
    """The first records of one committed (PHI-free) trace, values shortened."""
    f = min((run / "traces").glob("*/*.jsonl"))
    lines = f.read_text(encoding="utf-8").splitlines()[:9]
    out = []
    for ln in lines:
        r = json.loads(ln)
        out.append(json.dumps({k: r[k] for k in ("seq", "kind", "data") if k in r})[:230])
    return "\n".join(out)


# ---------------------------------------------------------------- sections

ARCH_SVG = """
<svg viewBox="0 0 980 432" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Arial" font-size="12">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker></defs>
<style>.b{fill:#f8fafc;stroke:#334155;stroke-width:1.2;rx:6}.m{fill:#ecfdf5;stroke:#047857}.p{fill:#fef2f2;stroke:#b91c1c}
.s{fill:#eff6ff;stroke:#1d4ed8}.t{font-weight:600}.n{fill:#475569;font-size:10.5px}.l{stroke:#475569;stroke-width:1.3;fill:none;marker-end:url(#a)}</style>
<rect class="b" x="10" y="20" width="120" height="58"/><text class="t" x="20" y="42">Input</text>
<text class="n" x="20" y="58">PNG JPEG PDF TIFF</text><text class="n" x="20" y="71">DICOM, zip bundle</text>
<rect class="b" x="160" y="20" width="120" height="58"/><text class="t" x="170" y="42">Router</text>
<text class="n" x="170" y="58">magic bytes; scan vs</text><text class="n" x="170" y="71">photo; corrupt</text>
<rect class="b" x="310" y="20" width="140" height="58"/><text class="t" x="320" y="42">Planner</text>
<text class="n" x="320" y="58">stage plan, budgets,</text><text class="n" x="320" y="71">retries, deadline</text>
<rect class="b" x="480" y="20" width="140" height="58"/><text class="t" x="490" y="42">Preprocess agent</text>
<text class="n" x="490" y="58">dewarp, flat-field, rotate,</text><text class="n" x="490" y="71">deskew, denoise (rules)</text>
<rect class="b p" x="650" y="20" width="150" height="58"/><text class="t" x="660" y="42">PHI agent</text>
<text class="n" x="660" y="58">anchors + regex + NER;</text><text class="n" x="660" y="71">mask before any model</text>
<rect class="b p" x="650" y="110" width="150" height="58"/><text class="t" x="660" y="132">DICOM de-id</text>
<text class="n" x="660" y="148">profile header actions,</text><text class="n" x="660" y="161">UID remap, burned-in</text>
<rect class="b m" x="830" y="20" width="140" height="74"/><text class="t" x="840" y="42">Extract</text>
<text class="n" x="840" y="58">OCR-first (default)</text><text class="n" x="840" y="71">hybrid: VLM re-reads</text><text class="n" x="840" y="84">low-conf crops</text>
<rect class="b" x="830" y="130" width="140" height="58"/><text class="t" x="840" y="152">Verify</text>
<text class="n" x="840" y="168">grounding, schema,</text><text class="n" x="840" y="181">cross-field, confidence</text>
<rect class="b p" x="830" y="220" width="140" height="58"/><text class="t" x="840" y="242">Injection check</text>
<text class="n" x="840" y="258">report text is data,</text><text class="n" x="840" y="271">never instructions</text>
<rect class="b" x="610" y="310" width="160" height="58"/><text class="t" x="620" y="332">Review queue</text>
<text class="n" x="620" y="348">low confidence, flags,</text><text class="n" x="620" y="361">injection, errors</text>
<rect class="b s" x="830" y="310" width="140" height="70"/><text class="t" x="840" y="332">Store + index</text>
<text class="n" x="840" y="348">SQLite JSON docs by</text><text class="n" x="840" y="361">subject_key; FTS5 BM25</text><text class="n" x="840" y="374">+ MiniLM embeddings</text>
<rect class="b s" x="540" y="395" width="430" height="30"/><text class="t" x="550" y="415">QA agent: RRF retrieval, extractive answer, verified citation, abstain</text>
<rect class="b m" x="300" y="130" width="300" height="74"/><text class="t" x="310" y="152">ModelClient (local, open source)</text>
<text class="n" x="310" y="168">Tesseract 5 (CPU) | Qwen2.5-VL-3B, Qwen2.5-3B (Ollama)</text>
<text class="n" x="310" y="181">Florence-2 crops | all-MiniLM-L6-v2 (ONNX)</text><text class="n" x="310" y="194">refuses any payload not PHI-masked</text>
<rect class="b" x="10" y="240" width="270" height="88"/><text class="t" x="20" y="262">Trace + eval</text>
<text class="n" x="20" y="278">PHI-free JSONL per case (ids, hashes, decisions)</text>
<text class="n" x="20" y="291">score records, offline re-score (models forbidden)</text>
<text class="n" x="20" y="304">VERSIONS.lock, CI gate on synthetic subset</text><text class="n" x="20" y="317">golden set run once (--final)</text>
<path class="l" d="M130,49 H160"/><path class="l" d="M280,49 H310"/><path class="l" d="M450,49 H480"/>
<path class="l" d="M620,49 H650"/><path class="l" d="M800,49 H830"/><path class="l" d="M380,78 V100 H725 V110"/>
<path class="l" d="M900,94 V130"/><path class="l" d="M900,188 V220"/><path class="l" d="M900,278 V310"/>
<path class="l" d="M860,278 V295 H690 V310"/><path class="l" d="M900,380 V395"/>
<path class="l" d="M800,139 H815 V110 H840"/><path class="l" d="M830,75 H815 V167 H600"/>
<path class="l" d="M380,78 V240" stroke-dasharray="4 3"/>
</svg>
"""


def tables_section(img: Path) -> str:
    """Section 11: PDF text layer + tables (after the golden run; dev and synthetic-set evidence only)."""
    te = ROOT / "eval/results/tables_eval.json"
    dc = ROOT / "eval/results/dev_tables_comparison.json"
    if not te.exists():
        return ""
    res = json.loads(te.read_text(encoding="utf-8"))
    conds = {**res.get("conditions_main", {}), **res["conditions"]}
    label = {"digital_main": "Digital PDF, main (before)", "digital": "Digital PDF, text layer (new)", "digital_ocr": "Digital PDF, OCR only (old)",
             "scan_clean": "Clean scan PDF (OCR + cell re-read)", "scan_poor": "Poor scan PDF (OCR + cell re-read)"}
    rows = []
    for k, c in conds.items():
        t, f, h = c["tables"], c["fields"], c["hidden_text"]
        rows.append([label.get(k, k), pct(t["detected"]), pct(t["shape_exact"]), f3(t["cell_exact_match"]), f3(t["cell_cer"]),
                     f3(f["mean_em"]), pct(h["flag_recall"]) if k.startswith("digital") else "-",
                     f"{c['seconds_per_doc']['p50']} s" if c.get("seconds_per_doc") else "-"])
    styles = sorted({s for c in conds.values() for s in c["tables"]["cell_em_by_style"]})
    style_rows = [[label.get(k, k), *[f3(c["tables"]["cell_em_by_style"].get(s)) for s in styles]] for k, c in conds.items()]
    dev_html = ""
    if dc.exists():
        d = json.loads(dc.read_text(encoding="utf-8"))
        dev_html = ("<h3>Dev set, main vs this change (218 documents, image-only inputs)</h3>"
                    + table(["Metric", "main", "with text layer + tables"],
                            [[r["metric"], r["main"], r["branch"]] for r in d["rows"]]))
    shot = ""
    over = next(iter(sorted((ROOT / "runs" / "inspect_tables").glob("*/25_tables/page_1_tables.png"))), None)
    if over:
        from PIL import Image, ImageOps

        im = Image.open(over).convert("RGB")
        box = ImageOps.invert(im.convert("L")).getbbox()  # crop the empty page margin
        if box:
            im = im.crop((max(0, box[0] - 20), max(0, box[1] - 20), min(im.width, box[2] + 20), min(im.height, box[3] + 20)))
        im.save(img / "tables_overlay.png")
        shot = ('<figure style="width:60%"><img src="img/tables_overlay.png" style="width:100%"/><figcaption>'
                'Step inspector, synthetic digital PDF: detected table cells (red).</figcaption></figure>')
    return f"""
<h2>11. Added after the golden run: PDF text layer and tables (dev and synthetic evidence)</h2>
<p>Digital PDFs were rasterised and OCR'd, and only a one-row exam table was parsed: a synthetic CT report with a nodule
table came out <i>ok</i> with the table garbled into Findings. Now the loader reads the PDF's own text layer (exact words
and boxes, invisible text dropped and sent to review as a prompt-injection risk), and a <code>tables</code> stage detects
ruled, rules-only and borderless tables, re-reads each ruled cell on OCR pages (single-line Tesseract on the cleaned cell
crop) and writes the better reading back for PHI masking, extraction and grounding. Table text no longer leaks into the
section prose; tables are emitted as an optional <code>tables</code> array (schema <code>tables.v1</code>; the locked
<code>cli_output.v1</code> contract is unchanged). The golden set was not re-run: these numbers are dev and a new
synthetic table set (40 documents x 3 variants, no IU text).</p>
{table(["Condition", "Tables found", "Exact shape", "Cell EM", "Cell CER", "Field EM", "Hidden text flagged", "p50 per doc"], rows, "small")}
{table(["Cell EM by style", *styles], style_rows, "small")}
{dev_html}
{shot}"""


def build(run: Path, out: Path, corpus: Path) -> Path:
    img = out / "img"
    img.mkdir(parents=True, exist_ok=True)
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))

    cmp_ = json.loads((ROOT / "eval/results/pipeline_comparison.json").read_text(encoding="utf-8"))
    attempts = (ROOT / "eval/results/attempts.md").read_text(encoding="utf-8")

    ov = overlays(run, corpus, img)
    ea_path = run / "error_analysis" / "summary.json"
    ea = json.loads(ea_path.read_text(encoding="utf-8")) if ea_path.exists() else None
    dicom_imgs = [dicom_before_after(run, img, d) for d in ("DICOM_001", "DICOM_020")]
    shots = {}
    insp = next(iter(sorted((ROOT / "runs" / "inspect_report").glob("*/index.html"))), None)
    if insp:
        screenshot(insp, img / "inspector.png", 1200, 1000)
        shots["inspector"] = "inspector.png"
    dash = ROOT / "docs" / "report" / "dashboard.png"
    if dash.exists():
        (img / "dashboard.png").write_bytes(dash.read_bytes())
        shots["dashboard"] = "dashboard.png"

    F = m["fields"]
    # --- results tables
    field_rows = []
    for f in FIELD_ORDER + ("mean",):
        o = F["overall"].get(f)
        if not o:
            continue
        row = [f"<b>{f}</b>" if f == "mean" else f, o.get("n", ""), f3(o["exact_match"]), f3(o["cer"]),
               pct(o["hallucination"]), pct(o["omission"])]
        row += [f3(F[t][f]["exact_match"]) if f in F.get(t, {}) else "-" for t in TIERS]
        field_rows.append(row)
    tier_rows = [[TIER_LABEL[t], m["pages_by_tier"].get(t, ""), f3(F[t]["mean"]["exact_match"]), f3(F[t]["mean"]["cer"]),
                  pct(F[t]["mean"]["hallucination"]), pct(F[t]["mean"]["omission"]),
                  pct(m["phi_documents"]["recall_by_tier"].get(t))] for t in TIERS]
    P = m["phi_documents"]
    phi_type_rows = [[k, pct(v)] for k, v in sorted(P["recall_by_type"].items(), key=lambda kv: kv[1])]
    resid = P.get("residual") or []
    from collections import Counter
    resid_by = Counter((r["type"], r["tier"]) for r in resid)
    resid_rows = [[t, TIER_LABEL[ti], n] for (t, ti), n in sorted(resid_by.items(), key=lambda kv: -kv[1])]
    D = m.get("dicom") or {}
    Q = m.get("qa") or {}
    I = m.get("injection") or {}
    R = m.get("review") or {}
    L = m.get("latency_ms_per_page") or {}

    cmp_rows = [[c["label"], f3(c["exact_match"]), f3(c["cer"]), pct(c["hallucination"]), pct(c["omission"]),
                 f3(c["em_clean_scan"]), f3(c["em_poor_scan"]), f3(c["em_phone_photo"]),
                 f"{c['latency_p50_s']} / {c['latency_p95_s']} s", f"{c['gpu_peak_mib']} MiB",
                 f"{c['vlm_calls_ok']} / {c['vlm_calls_failed']}"] for c in cmp_]

    ea_html = ""
    if ea:
        rows = [[k, w["doc_id"], w["page"], TIER_LABEL.get(w["tier"], w["tier"]), w["format"],
                 "yes" if w["handwriting"] else "no", w["mean_cer"] if w["mean_cer"] is not None else "-",
                 html.escape("; ".join(f"{f['field']}: {f['category']}" for f in w["failures"]))]
                for k, w in enumerate(ea["worst"], start=1)]
        cats = [[c, ea["categories_worst"].get(c, 0), n] for c, n in ea["categories_all_pages"].items()]
        ea_html = (table(["#", "Doc", "Page", "Tier", "Format", "Handwriting", "Mean CER", "Failures"], rows, "small")
                   + "<h3>Failure categories</h3>" + table(["Category", "Worst 10", "All scored pages"], cats))

    attempts_rows = []
    for ln in attempts.splitlines():
        if ln.startswith("| 2026"):
            c = [x.strip() for x in ln.strip("|").split("|")]
            attempts_rows.append([html.escape(x).replace("`", "") for x in c[1:]])

    git = subprocess.run(["git", "log", "--reverse", "--format=%ad|%h|%s", "--date=format:%d %b %H:%M"],
                         cwd=ROOT, capture_output=True, text=True, check=False).stdout.splitlines()
    chrono = [x.split("|", 2) for x in git if not x.split("|", 2)[2].startswith(("chore", "bd init"))]

    tv = meta.get("tool_versions") or {}
    tool_rows = [[k, v] for k, v in sorted(tv.items())]
    models = [
        ["OCR", "Tesseract 5 (LSTM, eng)", "CPU", "every pipeline"],
        ["Page / crop VLM", "Qwen2.5-VL-3B-Instruct (qwen2.5vl:3b, Q4_K_M, Ollama)", "GPU 4 GB, partly offloaded", "hybrid, VLM-first"],
        ["Crop fallback", "Florence-2-base", "GPU/CPU", "hybrid when Ollama is down"],
        ["Answer LLM", "Qwen2.5-3B-Instruct (qwen2.5:3b, Ollama)", "GPU", "optional; default QA is extractive"],
        ["Embedder", "sentence-transformers/all-MiniLM-L6-v2 (fastembed ONNX)", "CPU", "retrieval"],
        ["PHI NER", "Presidio + spaCy when available, else honorific rules", "CPU", "PHI agent"],
    ]
    prompts = sorted((ROOT / "prompts").glob("*.md"))

    tables_html = tables_section(img)

    def img_tag(name, cap, w="100%"):
        return f'<figure style="width:{w}"><img src="img/{name}" style="width:100%"/><figcaption>{cap}</figcaption></figure>'

    ov_html = "".join(img_tag(n, f"{lab} page: {p['doc_id']} p{p['page']} ({TIER_LABEL.get(p['tier'])}, {p['format']}), "
                               f"mean CER {p['mean_cer']}, badness {p['badness']}", "32%") for lab, n, p in ov)

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>radintake report</title>
<style>
@page {{ size: A4; margin: 16mm 14mm; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; color: #0f172a; line-height: 1.42; }}
h1 {{ font-size: 21pt; margin: 0 0 4px; }} h2 {{ font-size: 14pt; margin: 22px 0 6px; border-bottom: 1px solid #cbd5e1;
 padding-bottom: 3px; break-after: avoid; }} h3 {{ font-size: 11.5pt; margin: 14px 0 4px; break-after: avoid; }}
table {{ border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 9pt; break-inside: auto; }}
th, td {{ border: 1px solid #cbd5e1; padding: 3px 5px; text-align: left; vertical-align: top; }}
th {{ background: #f1f5f9; }} tr {{ break-inside: avoid; }} table.small {{ font-size: 8pt; }}
.kpi {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin: 8px 0; }}
.kpi div {{ border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 8px; }} .kpi b {{ display: block; font-size: 15pt; }}
.kpi span {{ font-size: 8.5pt; color: #475569; }} figure {{ margin: 6px 0; display: inline-block; vertical-align: top; }}
figcaption {{ font-size: 8.5pt; color: #475569; }} .row figure {{ margin-right: 1%; }}
pre {{ font-size: 7.5pt; background: #f8fafc; border: 1px solid #e2e8f0; padding: 6px; white-space: pre-wrap; word-break: break-all; }}
.note {{ font-size: 9pt; color: #475569; }} .sub {{ color: #475569; margin-bottom: 10px; }}
</style></head><body>
<h1>radintake: PHI-safe intake of radiology reports and DICOM</h1>
<div class="sub">Senior AI Engineer assignment (OCR / VLM). Repository: github.com/varmakvl/radintake, branch
sprint/radintake-v1 at {meta.get('git_sha', '')[:7]}. Golden run {html.escape(meta.get('run_id', ''))}.
All models open source and local: cost $0 per page.</div>

<div class="kpi">
<div><b>{f3(F['overall']['mean']['exact_match'])}</b><span>field exact match, golden ({m['n_docs']} docs, {m['n_pages']} pages)</span></div>
<div><b>{pct(P['recall'])}</b><span>document PHI recall (precision {pct(P['precision'])})</span></div>
<div><b>{pct(D.get('header_phi_recall'))}</b><span>DICOM header PHI recall; valid after redaction {pct(D.get('valid'))}</span></div>
<div><b>{f3(Q.get('recall_at_5'))}</b><span>retrieval recall@5; citations {pct(Q.get('citation_accuracy'))} accurate</span></div>
<div><b>{f3(F['clean_scan']['mean']['exact_match'])}</b><span>exact match, clean scans</span></div>
<div><b>{f3(F['phone_photo']['mean']['exact_match'])}</b><span>exact match, phone photos</span></div>
<div><b>{(L.get('p50') or 0) / 1000:.1f} s</b><span>latency p50 per page (p95 {(L.get('p95') or 0) / 1000:.1f} s), CPU only</span></div>
<div><b>{pct(R.get('docs_needs_review'))}</b><span>documents sent to human review</span></div>
</div>

<h2>1. Architecture</h2>
<p>A fixed pipeline with agentic branches. Fixed stages are where correctness must be auditable (routing, PHI masking,
DICOM de-identification, schema validation); agents decide where judgement pays off (which preprocessing to apply, when
to ask the VLM to re-read a crop, whether a value is grounded, whether text is an injected instruction, whether to
answer or abstain). Every decision and its numbers go into a PHI-free trace. PHI masking happens <b>before</b> any
model call, enforced in <code>ModelClient</code>, and every box is kept in original page pixels through the
preprocessing homographies, so provenance always points at the page the user uploaded.</p>
{ARCH_SVG}
<p class="note">Storage: one SQLite file with JSON documents keyed by a pseudonymous <code>subject_key</code> (patient and
uploader are assumed to be the same person); FTS5 and in-row embeddings serve retrieval. A vector database was not
needed at this scale (hundreds of documents) and would add a service to secure.</p>

<h2>2. Data</h2>
<p>330 IU Chest X-ray reports (Open-i, CC BY-NC-ND) rendered by Playwright at 200 DPI into three hospital templates with
synthetic identities (Faker, seed 140), 82 multi-page documents, handwriting on 40% of documents (4 OFL/Apache fonts) and
11+ pages with injected instructions. Each document is degraded into one tier (clean scan, poor scan, phone photo) with
one composed homography, so every ground-truth box stays exact, and written as PNG, JPEG, multi-page PDF or multi-page
TIFF. Exact and MinHash near-duplicates are grouped and never cross the split; dev has 218 documents and the frozen
golden set {m['n_docs']} documents / {m['n_pages']} pages (hash-locked, scored once). DICOM ground truth comes from TCIA
Pseudo-PHI-DICOM-Data, by pairing each evaluation file with its curated twin through the UID crosswalk; burned-in text
is scored against text-pixel masks. Nothing licensed is committed: scripts, seeds, ids and hashes only.</p>

<h2>3. Golden-set results (OCR-first, the default pipeline)</h2>
<h3>Per field, overall and per tier</h3>
{table(["Field", "n", "EM", "CER", "Halluc.", "Omission", "EM clean", "EM poor", "EM photo"], field_rows)}
<h3>Per tier</h3>
{table(["Tier", "Pages", "Mean EM", "Mean CER", "Hallucination", "Omission", "PHI recall"], tier_rows)}
<p class="note">EM is exact match after whitespace collapse and casefold; CER = Levenshtein / len(GT). Hallucination =
non-empty value with partial-ratio &lt; 80 against all printed field text; omission = GT present, value empty.
Handwritten fields: n={m['handwritten_fields']['n']}, EM {f3(m['handwritten_fields']['exact_match'])},
CER {f3(m['handwritten_fields']['cer'])}.</p>

<h3>Retrieval and QA ({Q.get('n_questions')} golden questions, MeSH-labelled)</h3>
{table(["recall@5", "Citation accuracy", "Answer correctness", "Correct abstention", "Unanswerable abstained", "False abstention", "Injection probes correct"],
       [[f3(Q.get('recall_at_5')), pct(Q.get('citation_accuracy')), pct(Q.get('answer_correctness')), pct(Q.get('correct_abstention')),
         pct(Q.get('unanswerable_abstained')), pct(Q.get('answerable_false_abstention')), pct(Q.get('injection_probes_correct'))]])}

<h3>Operations</h3>
{table(["Latency p50 / p95 per page", "GPU peak", "Cost per page", "Docs to review", "Fields to review", "Model calls"],
       [[f"{(L.get('p50') or 0) / 1000:.2f} / {(L.get('p95') or 0) / 1000:.2f} s", f"{m.get('gpu_peak_mib', 0)} MiB",
         f"${m.get('cost_usd_per_page', 0):.2f}", pct(R.get('docs_needs_review')), pct(R.get('fields_needs_review')), m.get('model_calls', 0)]])}

<h2>4. Pipeline comparison (dev, 18 documents, 6 per tier)</h2>
{table(["Pipeline", "EM", "CER", "Halluc.", "Omission", "EM clean", "EM poor", "EM photo", "p50 / p95 per doc", "GPU peak", "VLM calls ok / failed"], cmp_rows, "small")}
<p>OCR-first is the default. Hybrid (Qwen2.5-VL-3B re-reads only low-confidence sections, as masked crops) gains 0.9
points, all on phone photos, at 5x the latency. VLM-first loses on every tier: the 3B model, at the 1100 px a 4 GB GPU
allows, drops or paraphrases long sections (omission 34%). On a larger GPU, hybrid would be the first thing to
re-evaluate. The comparison uses the dev set only; the golden set is scored once with the chosen pipeline.</p>

<h2>5. PHI approach and residual cases</h2>
<p><b>Documents.</b> Three layers on OCR words in original coordinates: (1) layout anchors (a label such as
"Patient Name:" takes the tokens to its right up to the next label or a column gap), (2) regex over IDs, accession
formats and dates, (3) NER for names in free text plus a credential rule for clinician sign-offs ("Dr X, MD",
"RT(R) initials"). The masked page is the only thing a model sees; originals and the value vault stay local.
<b>DICOM.</b> A Basic-Profile-style action table (remove / replace / keep, private tags, overlay 60xx, curve data,
device ids), consistent UID remapping, date fail-safes and line-level burned-in text redaction, then re-validation.</p>
{table(["PHI type", "Recall (golden)"], phi_type_rows)}
<p>Residual (undetected) document PHI on golden: <b>{len(resid)}</b> of {P.get('n_gt')} instances. Every residual case is
listed by id in <code>metrics.json</code>; grouped here by type and tier:</p>
{table(["Type", "Tier", "Missed"], resid_rows, "small")}
<p class="note">Technologist and radiologist names in handwriting or signature blocks are the weakest types; patient-name
misses concentrate on photos where OCR does not read the header at all (see error analysis). A document is never
released with an unresolved PHI-risk flag: such documents go to review.</p>
<h3>DICOM</h3>
{table(["Files", "Valid after redaction", "Header PHI recall", "UIDs remapped", "Burned-in box recall", "Text-pixel recall", "Pixel precision"],
       [[D.get('n'), pct(D.get('valid')), pct(D.get('header_phi_recall'), 2), pct(D.get('uids_remapped')), pct(D.get('burned_in_recall')),
         pct(D.get('burned_in_text_pixel_recall')), pct(D.get('burned_in_pixel_precision'))]])}
<p class="note">The 48 TCIA files are shared by dev and golden runs (the collection is small), so DICOM numbers are not
held out. Residual DICOM tags by file are listed in <code>metrics.json</code>.</p>
<div class="row">{''.join(img_tag(n, f'{n[6:-4]}: original (left) and de-identified (right); red box = burned-in text region, zoomed below (original above, de-identified below)', '49%') for n in dicom_imgs)}</div>

<h2>6. Prompt-injection results</h2>
{table(["Injected golden docs", "Flagged", "Changed behaviour", "False flags on clean docs"],
       [[I.get('n_injected_docs'), I.get('flagged'), I.get('changed_behaviour'), I.get('false_flags_on_clean_docs')]])}
<p>Report text is treated as data: extraction is extractive (OCR-first) or schema-constrained (VLM), the injection agent
flags imperative and role-play patterns, and the QA agent only quotes verified text. "Changed behaviour" means the
impression echoed the injected text, became "normal" against a non-normal truth, or a section leaked the patient name.
On golden only 1 of the 4 injected documents was flagged (dev: 8 of 11), but none changed behaviour: the defence that
matters is structural (extraction never follows document text), and the detector is a review signal whose recall drops when
OCR garbles the injected sentence on degraded pages. Golden QA injection probes: {pct(Q.get('injection_probes_correct'))} correct
(all 4 probes found and cited the injected study; 2 quoted an impression that does not match the true one, because the
extracted impression was wrong on that degraded page; the QA agent only quotes verified text, so it cannot carry out an
injected instruction, but it can quote a wrong extraction).</p>

<h2>7. Error analysis: the 10 worst golden pages</h2>
<p>Badness = mean field CER on the page (omission = 1) + 0.25 per missed PHI instance. One category per failed field.</p>
{ea_html}
<h3>Best, median and worst page</h3>
<p class="note">Green: ground truth. Blue: prediction provenance (purple if from the VLM). Red: missed PHI. Boxes are
drawn on the page as uploaded (original pixels), so a page scanned upside down is shown upside down.</p>
<div class="row">{ov_html}</div>

<h2>8. What was done, in order (including failed attempts)</h2>
{table(["When", "Commit", "Step"], [[a, f"<code>{b}</code>", html.escape(c)] for a, b, c in chrono], "small")}
<h3>Tried and rejected or kept (dev evidence)</h3>
{table(["Attempt", "Evidence", "Decision"], attempts_rows, "small")}
<p class="note">CI regression gate: commit <code>f5d4033</code> misspells the patient-name anchor on purpose; CI goes red
(unit tests and, locally, the gate: PHI recall 0.96 to 0.75), and <code>19c9d6e</code> reverts it (green). A known
limit: the 6-page synthetic gate subset did not catch a heading-threshold regression on its own.</p>

<h2>9. Technical details</h2>
{table(["Role", "Model", "Runs on", "Used by"], models)}
<p><b>Hardware:</b> {html.escape(platform.platform())}, NVIDIA RTX 3050 Laptop 4 GB, Python {platform.python_version()}.
<b>Seeds:</b> data and split 140. <b>Prompts</b> (immutable, hash-locked in <code>eval/VERSIONS.lock</code>):
{', '.join(p.stem for p in prompts)}. <b>Schemas:</b> report.v1, cli_output.v1, trace.v1.
<b>Cost:</b> $0 per page (no API; electricity not counted).</p>
{table(["Library", "Version"], tool_rows, "small")}

<h2>10. Traces</h2>
<p>Each case writes a JSONL trace with ids, hashes, decisions, numbers and timings, never values. The golden traces are
committed under <code>traces/final/</code> and re-score offline with every model call forbidden
(<code>python -m radintake.eval.rescore --traces traces/final --check</code>). First records of one golden trace:</p>
<pre>{html.escape(trace_excerpt(run))}</pre>
{img_tag(shots['inspector'], 'Step inspector on a synthetic fixture: each stage input and output, masked page, OCR words, extraction with provenance.') if 'inspector' in shots else ''}
{img_tag(shots['dashboard'], 'Progress dashboard (Claude artifact) built from the run records.') if 'dashboard' in shots else ''}
{tables_html}
</body></html>"""
    page = out / "report.html"
    page.write_text(doc, encoding="utf-8")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(page.resolve().as_uri())
        pg.wait_for_load_state("networkidle")
        pg.pdf(path=str(out / "report.pdf"), format="A4", print_background=True,
               margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
               display_header_footer=True, header_template="<span></span>",
               footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#64748b">'
                               'radintake report - <span class="pageNumber"></span> / <span class="totalPages"></span></div>')
        b.close()
    return out / "report.pdf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "report" / "out")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus")
    a = ap.parse_args()
    print(build(a.run, a.out, a.corpus))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
