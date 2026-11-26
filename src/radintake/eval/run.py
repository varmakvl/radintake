"""Run the pipeline on a split and write a self-contained, re-scorable run record.

    python -m radintake.eval.run --split dev [--pipeline ocr_first|vlm_first|hybrid]
                                 [--sample-per-tier K] [--limit N] [--dicom] [--out runs/<name>]
    python -m radintake.eval.run --split golden --final      # the ONE golden run (guarded)
    python -m radintake.eval.run --split golden_v2 --final   # the ONE golden v2 run (guarded)
    python -m radintake.eval.run --split golden_v2 --check-only   # lock + GT check, runs nothing

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

Golden v2 (``splits/golden_v2_ids.txt``, built by ``data_prep.heldout``, corpus ``data/corpus_v2``)
has its own guard, ``eval/GOLDEN_V2_USED.json``, written by its first ``--final`` run, and is only
run when ``sha256(golden_v2_ids.txt)`` matches ``splits/golden_v2.lock``. ``--check-only`` verifies
the lock, the guard and the GT presence without running the pipeline or touching the guard.
"""

from __future__ import annotations

import argparse
import hashlib
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
from radintake.eval.metrics import aggregate, score_dicom, score_document, score_qa

SPLITS = REPO_ROOT / "splits"
GOLDEN_USED = REPO_ROOT / "eval" / "GOLDEN_USED.json"
GOLDEN_V2_USED = REPO_ROOT / "eval" / "GOLDEN_V2_USED.json"
GOLDEN_V2_CORPUS = Path("data/corpus_v2")
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


def lock_hash(path: Path) -> str:
    """sha256 of a split ids file with LF line endings (same rule as data_prep.split.golden_hash)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_golden_v2(corpus: Path, splits: Path | None = None) -> dict[str, Any]:
    """Validate the golden v2 lock and GT presence; never runs the pipeline, never writes the guard.

    ``ok`` is the lock verdict; ``gt_missing`` lists ids without ``<corpus>/gt/<id>.json`` (``None``
    when the corpus directory is absent, e.g. in CI)."""
    splits = splits or SPLITS
    ids_path, lock_path = splits / "golden_v2_ids.txt", splits / "golden_v2.lock"
    if not ids_path.is_file() or not lock_path.is_file():
        return {"ok": False, "error": f"missing {ids_path.name} or {lock_path.name}"}
    actual, expected = lock_hash(ids_path), lock_path.read_text(encoding="utf-8").strip()
    ids = [ln.split("\t")[0] for ln in ids_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    gt_missing = None
    if (corpus / "gt").is_dir():
        gt_missing = [i for i in ids if not (corpus / "gt" / f"{i}.json").is_file()]
    return {"ok": actual == expected, "lock": expected, "actual": actual, "n_ids": len(ids),
            "corpus": str(corpus), "gt_missing": gt_missing, "used": GOLDEN_V2_USED.exists()}


def sample(ids: list[str], gts: dict[str, dict], per_tier: int, seed: int = SEED) -> list[str]:
    rng = np.random.default_rng(seed)
    out = []
    for tier in ("clean_scan", "poor_scan", "phone_photo"):
        pool = sorted(i for i in ids if gts[i]["tier"] == tier)
        out += sorted(rng.choice(pool, size=min(per_tier, len(pool)), replace=False).tolist())
    return sorted(out)


def qa_on_run(run_dir: Path, qa_path: Path, gts: dict[str, dict], preds: dict[str, dict] | None = None) -> list[dict]:
    """Index a run's document outputs (case id = doc id) and score every question of ``qa_path``."""
    from radintake.agents.qa import answer
    from radintake.retrieval import RetrievalIndex

    if preds is None:
        preds = {}
        for d, g in gts.items():
            p = run_dir / "outputs" / f"{g['file']}.json"
            if p.is_file():
                preds[d] = json.loads(p.read_text(encoding="utf-8"))
    db = run_dir / "retrieval.sqlite"
    db.unlink(missing_ok=True)
    index = RetrievalIndex(db, config=load_config())
    for d, pred in preds.items():
        if pred.get("status") != "error" and pred.get("report"):
            index.add_case(d, pred)
    qs = [json.loads(ln) for ln in qa_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    recs = []
    for q in qs:
        t0 = time.perf_counter()
        res = answer(q["question"], index)
        rec = score_qa(q, res, gts)
        rec["ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        recs.append(rec)
    return recs


def run(*, split: str, out: Path, corpus: Path, pipeline: str | None = None, per_tier: int = 0, limit: int = 0,
        dicom_dir: Path | None = None, docs_gt: dict[str, dict] | None = None, files_dir: Path | None = None,
        label: str | None = None, qa_path: Path | None = None) -> dict[str, Any]:
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
    preds: dict[str, dict] = {}
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
            preds[i] = pred
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
    if qa_path is not None and qa_path.is_file():
        scores += qa_on_run(out, qa_path, {i: docs_gt[i] for i in ids}, preds)
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
    ap.add_argument("--split", choices=("dev", "golden", "golden_v2"), required=True)
    ap.add_argument("--pipeline", choices=("ocr_first", "vlm_first", "hybrid"))
    ap.add_argument("--sample-per-tier", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dicom", action="store_true", help="also evaluate data/dicom_gt")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="corpus directory (default data/corpus; data/corpus_v2 for golden_v2)")
    ap.add_argument("--dicom-dir", type=Path, default=Path("data/dicom_gt"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--final", action="store_true", help="required for the single golden / golden_v2 run")
    ap.add_argument("--check-only", action="store_true",
                    help="golden_v2: verify the lock and GT presence; run nothing, write nothing")
    ap.add_argument("--qa", type=Path, default=None,
                    help="QA set to score (default: eval/golden_qa.jsonl for golden, eval/dev_qa.jsonl for dev)")
    ap.add_argument("--qa-only", type=Path, default=None, metavar="RUN_DIR",
                    help="score the QA set on an existing run's outputs (no pipeline re-run) and update its record")
    a = ap.parse_args(argv)
    if a.corpus is None:
        a.corpus = GOLDEN_V2_CORPUS if a.split == "golden_v2" else Path("data/corpus")
    if a.check_only:
        if a.split != "golden_v2":
            ap.error("--check-only applies to --split golden_v2")
        res = check_golden_v2(a.corpus)
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] and not res.get("gt_missing") else 1
    if a.split == "golden_v2":
        if not a.final:
            ap.error("the golden_v2 split is only scored by the final run: pass --final")
        if GOLDEN_V2_USED.exists():
            ap.error(f"golden v2 set already used: {GOLDEN_V2_USED.read_text(encoding='utf-8').strip()}")
        res = check_golden_v2(a.corpus)
        if not res["ok"]:
            ap.error(f"golden_v2 lock mismatch: {res}")
        if res.get("gt_missing"):
            ap.error(f"golden_v2 ground truth missing for {len(res['gt_missing'])} docs under {a.corpus}")
    if a.split == "golden":
        if not a.final:
            ap.error("the golden split is only scored by the final run: pass --final")
        if GOLDEN_USED.exists():
            ap.error(f"golden set already used: {GOLDEN_USED.read_text(encoding='utf-8').strip()}")
    qa_default = {"golden": "golden_qa.jsonl", "golden_v2": "golden_v2_qa.jsonl"}.get(a.split, "dev_qa.jsonl")
    qa_path = a.qa or (REPO_ROOT / "eval" / qa_default)
    if a.qa_only:
        rd = a.qa_only
        meta = json.loads((rd / "run.json").read_text(encoding="utf-8"))
        gts = {i: json.loads((a.corpus / "gt" / f"{i}.json").read_text(encoding="utf-8")) for i in meta["doc_ids"]}
        old = [json.loads(ln) for ln in (rd / "scores.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        scores = [r for r in old if r["kind"] != "qa"] + qa_on_run(rd, qa_path, gts)
        (rd / "scores.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scores), encoding="utf-8")
        timings = json.loads((rd / "timings.json").read_text(encoding="utf-8"))
        m = aggregate(scores, timings)
        (rd / "metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
        print(json.dumps(m.get("qa"), indent=1))
        return 0
    out = a.out or Path("runs") / f"{a.split}_{a.pipeline or 'default'}{'_s' + str(a.sample_per_tier) if a.sample_per_tier else ''}"
    m = run(split=a.split, out=out, corpus=a.corpus, pipeline=a.pipeline, per_tier=a.sample_per_tier, limit=a.limit,
            dicom_dir=a.dicom_dir if a.dicom else None, qa_path=qa_path)
    if a.split == "golden":
        GOLDEN_USED.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_USED.write_text(json.dumps({"run": str(out), "run_id": json.loads((out / "run.json").read_text())["run_id"],
                                           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n", encoding="utf-8")
    if a.split == "golden_v2":
        GOLDEN_V2_USED.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_V2_USED.write_text(json.dumps({"run": str(out), "run_id": json.loads((out / "run.json").read_text())["run_id"],
                                              "lock": check_golden_v2(a.corpus)["lock"],
                                              "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n",
                                  encoding="utf-8")
    print(json.dumps({"out": str(out), "mean": m["fields"]["overall"]["mean"], "phi_recall": m["phi_documents"]["recall"],
                      "latency": m.get("latency_ms_per_page"), "dicom": {k: v for k, v in (m.get("dicom") or {}).items() if k != "residual"}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
