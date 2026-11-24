"""Run the pipeline on a split and write a self-contained, re-scorable run record.

    python -m radintake.eval.run --split dev [--pipeline ocr_first|vlm_first|hybrid]
                                 [--sample-per-tier K] [--limit N] [--dicom] [--out runs/<name>]
    python -m radintake.eval.run --split golden --final      # the ONE golden run (guarded)

Run record (``--out``)::

    outputs/            pipeline outputs (values + PHI; local only - runs/ is gitignored)
    traces/<run_id>/    PHI-free traces (hashes, model ids, prompt hashes, timings)
    scores.jsonl        PHI-free score records (radintake.eval.metrics) - the re-scoring input
    timings.json        per-case wall time + pages, GPU peak
    run.json            split, pipeline, sample, git sha, model ids, prompt hashes
    metrics.json        aggregate() of scores + timings

Everything except ``outputs/`` can be committed (final run: ``traces/final/``). The golden split is
guarded: the first ``--final`` golden run writes ``eval/GOLDEN_USED.json``; later golden runs are
refused (the brief: use the golden set once for the final numbers).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from radintake.config import REPO_ROOT, load_config
from radintake.eval.metrics import aggregate, score_dicom, score_document

SPLITS = REPO_ROOT / "splits"
GOLDEN_USED = REPO_ROOT / "eval" / "GOLDEN_USED.json"
SEED = 140


class GpuSampler:
    """Polls nvidia-smi for used GPU memory; peak in MiB (None without an NVIDIA GPU)."""

    def __init__(self, period_s: float = 0.5):
        self.period_s, self.peak, self.baseline = period_s, None, None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _read(self) -> int | None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5, check=False)
            return int(out.stdout.split()[0]) if out.returncode == 0 and out.stdout.strip() else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def _loop(self):
        while not self._stop.is_set():
            v = self._read()
            if v is not None:
                self.peak = v if self.peak is None else max(self.peak, v)
            self._stop.wait(self.period_s)

    def __enter__(self):
        self.baseline = self._read()
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=5)


def split_ids(name: str) -> list[str]:
    return [ln.split("\t")[0] for ln in (SPLITS / f"{name}_ids.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]


def sample(ids: list[str], gts: dict[str, dict], per_tier: int, seed: int = SEED) -> list[str]:
    rng = np.random.default_rng(seed)
    out = []
    for tier in ("clean_scan", "poor_scan", "phone_photo"):
        pool = sorted(i for i in ids if gts[i]["tier"] == tier)
        out += sorted(rng.choice(pool, size=min(per_tier, len(pool)), replace=False).tolist())
    return sorted(out)


def run(*, split: str, out: Path, corpus: Path, pipeline: str | None = None, per_tier: int = 0, limit: int = 0,
        dicom_dir: Path | None = None, docs_gt: dict[str, dict] | None = None, files_dir: Path | None = None,
        label: str | None = None) -> dict[str, Any]:
    from radintake.pipeline import process_file
    from radintake.tracing.writer import TraceWriter

    if pipeline:
        os.environ["RADINTAKE_EXTRACTION_PIPELINE"] = pipeline
    cfg = load_config()
    if docs_gt is None:
        ids = split_ids(split)
        docs_gt = {i: json.loads((corpus / "gt" / f"{i}.json").read_text(encoding="utf-8")) for i in ids}
        files_dir = corpus / "files"
    ids = sorted(docs_gt)
    if per_tier:
        ids = sample(ids, docs_gt, per_tier)
    if limit:
        ids = ids[:limit]
    if out.exists():
        shutil.rmtree(out)
    (out / "outputs").mkdir(parents=True)
    tw = TraceWriter(out / "traces", model_ids=cfg.model_ids, offline=cfg.offline)
    scores: list[dict] = []
    cases: list[dict] = []
    dicom_gt = []
    if dicom_dir is not None:
        dicom_gt = [json.loads(ln) for ln in (dicom_dir / "gt.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    t_run = time.perf_counter()
    with GpuSampler() as gpu:
        for i in ids:
            gt = docs_gt[i]
            path = files_dir / gt["file"]
            t0 = time.perf_counter()
            pred = process_file(path, out / "outputs", config=cfg, trace_writer=tw)
            ms = (time.perf_counter() - t0) * 1000.0
            cases.append({"id": i, "ms": round(ms, 1), "pages": gt.get("n_pages", 1), "tier": gt["tier"]})
            scores += score_document(pred, gt)
        for g in dicom_gt:
            path = dicom_dir / "inputs" / f"{g['dicom_id']}.dcm"
            t0 = time.perf_counter()
            pred = process_file(path, out / "outputs", config=cfg, trace_writer=tw)
            cases.append({"id": g["dicom_id"], "ms": round((time.perf_counter() - t0) * 1000.0, 1), "pages": 1,
                          "tier": "dicom"})
            scores.append(score_dicom(pred, g, path, out / "outputs"))
    tw.finalize()
    model_calls = sum(ln.count('"kind": "model_call"') + ln.count('"kind":"model_call"')
                      for p in (out / "traces").rglob("*.jsonl") for ln in p.read_text(encoding="utf-8").splitlines())
    doc_cases = [c for c in cases if c["tier"] != "dicom"]
    timings = {"cases": doc_cases, "dicom_cases": [c for c in cases if c["tier"] == "dicom"],
               "gpu_peak_mib": gpu.peak, "gpu_baseline_mib": gpu.baseline, "model_calls": model_calls,
               "wall_s": round(time.perf_counter() - t_run, 1)}
    manifest = json.loads(tw.manifest_path.read_text(encoding="utf-8"))
    meta = {"label": label or out.name, "split": split, "pipeline": pipeline or cfg.raw.get("extraction", {}).get("pipeline"),
            "n_docs": len(ids), "n_dicom": len(dicom_gt), "sample_per_tier": per_tier, "sample_seed": SEED,
            "doc_ids": ids, "run_id": tw.run_id, "git_sha": manifest.get("git_sha"),
            "models": manifest.get("models"), "models_called": manifest.get("models_called"),
            "prompts": manifest.get("prompts"), "tool_versions": manifest.get("tool_versions"),
            "offline": cfg.offline}
    (out / "scores.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scores), encoding="utf-8")
    (out / "timings.json").write_text(json.dumps(timings, indent=1), encoding="utf-8")
    (out / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    metrics = aggregate(scores, timings)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m radintake.eval.run")
    ap.add_argument("--split", choices=("dev", "golden"), required=True)
    ap.add_argument("--pipeline", choices=("ocr_first", "vlm_first", "hybrid"))
    ap.add_argument("--sample-per-tier", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dicom", action="store_true", help="also evaluate data/dicom_gt")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--dicom-dir", type=Path, default=Path("data/dicom_gt"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--final", action="store_true", help="required for the single golden run")
    a = ap.parse_args(argv)
    if a.split == "golden":
        if not a.final:
            ap.error("the golden split is only scored by the final run: pass --final")
        if GOLDEN_USED.exists():
            ap.error(f"golden set already used: {GOLDEN_USED.read_text(encoding='utf-8').strip()}")
    out = a.out or Path("runs") / f"{a.split}_{a.pipeline or 'default'}{'_s' + str(a.sample_per_tier) if a.sample_per_tier else ''}"
    m = run(split=a.split, out=out, corpus=a.corpus, pipeline=a.pipeline, per_tier=a.sample_per_tier, limit=a.limit,
            dicom_dir=a.dicom_dir if a.dicom else None)
    if a.split == "golden":
        GOLDEN_USED.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_USED.write_text(json.dumps({"run": str(out), "run_id": json.loads((out / "run.json").read_text())["run_id"],
                                           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "mean": m["fields"]["overall"]["mean"], "phi_recall": m["phi_documents"]["recall"],
                      "latency": m.get("latency_ms_per_page"), "dicom": {k: v for k, v in (m.get("dicom") or {}).items() if k != "residual"}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
