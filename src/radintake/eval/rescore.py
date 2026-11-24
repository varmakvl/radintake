"""Re-score a past run from its record, with every model call forbidden.

    python -m radintake.eval.rescore --traces <run dir | traces dir> [--offline] [--check] [--verify-gt]

* With a run record (``scores.jsonl`` next to or above the traces): recompute every metric from the
  PHI-free score records + timings and write ``metrics.rescored.json``. ``--check`` exits 1 when the
  result differs from the stored ``metrics.json``. ``--verify-gt`` recomputes each field's
  ground-truth hash from the local corpus GT and checks it matches the hash in the record (proves the
  scores were made against this ground truth).
* With traces only (e.g. a batch CLI run): summarise them - cases, statuses, latency, model calls,
  model ids and prompt hashes.

``RADINTAKE_FORBID_MODEL_CALLS=1`` is set before anything else is imported, so any accidental model
call raises instead of running.
"""

from __future__ import annotations

import os

os.environ["RADINTAKE_FORBID_MODEL_CALLS"] = "1"  # before any radintake import that could call a model

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from radintake.eval.metrics import _pct, aggregate, norm, value_hash


def find_run_dir(p: Path) -> Path | None:
    for cand in (p, p.parent, p.parent.parent):
        if (cand / "scores.jsonl").is_file():
            return cand
    return None


def summarise_traces(trace_dir: Path) -> dict:
    manifests = sorted(trace_dir.rglob("manifest.json"))
    cases, statuses, models, prompts, calls = [], Counter(), set(), {}, 0
    for m in manifests:
        man = json.loads(m.read_text(encoding="utf-8"))
        for c in man.get("cases", []):
            cases.append(c.get("duration_ms"))
            statuses[c.get("status")] += 1
        models |= set(man.get("models_called") or [])
        prompts.update(man.get("prompts") or {})
    for f in trace_dir.rglob("*.jsonl"):
        for ln in f.read_text(encoding="utf-8").splitlines():
            if ln.strip() and json.loads(ln).get("kind") == "model_call":
                calls += 1
    ms = [c for c in cases if c is not None]
    return {"runs": len(manifests), "cases": len(cases), "status": dict(statuses), "model_calls": calls,
            "models_called": sorted(models), "prompts": prompts,
            "latency_ms_per_case": {"p50": _pct(ms, 0.5), "p95": _pct(ms, 0.95)}}


def verify_gt(run_dir: Path, scores: list[dict], corpus: Path) -> list[str]:
    problems = []
    cache: dict[str, dict] = {}
    for r in scores:
        if r["kind"] != "field" or not r.get("gt_hash"):
            continue
        gtp = corpus / "gt" / f"{r['doc_id']}.json"
        if not gtp.is_file():
            return [f"ground truth not available locally ({gtp}); skipped"]
        gt = cache.setdefault(r["doc_id"], json.loads(gtp.read_text(encoding="utf-8")))
        if value_hash(norm(gt["expected"].get(r["field"]))) != r["gt_hash"]:
            problems.append(f"{r['doc_id']}:{r['field']}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m radintake.eval.rescore")
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--offline", action="store_true", help="accepted for clarity; re-scoring never calls a model")
    ap.add_argument("--check", action="store_true", help="exit 1 if the re-scored metrics differ from metrics.json")
    ap.add_argument("--verify-gt", action="store_true")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    a = ap.parse_args(argv)
    if not a.traces.exists():
        print(f"not found: {a.traces}", file=sys.stderr)
        return 2
    run_dir = find_run_dir(a.traces)
    if run_dir is None:
        print(json.dumps(summarise_traces(a.traces), indent=1))
        return 0
    scores = [json.loads(ln) for ln in (run_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    timings = json.loads((run_dir / "timings.json").read_text(encoding="utf-8")) if (run_dir / "timings.json").is_file() else None
    metrics = aggregate(scores, timings)
    (run_dir / "metrics.rescored.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    rc = 0
    stored = run_dir / "metrics.json"
    same = None
    if stored.is_file():
        same = json.loads(stored.read_text(encoding="utf-8")) == json.loads(json.dumps(metrics))
        if a.check and not same:
            rc = 1
    gt_problems = verify_gt(run_dir, scores, a.corpus) if a.verify_gt else []
    if a.verify_gt and gt_problems and not gt_problems[0].startswith("ground truth not available"):
        rc = 1
    print(json.dumps({"run": str(run_dir), "records": len(scores), "matches_stored_metrics": same,
                      "gt_hash_problems": gt_problems[:20], "mean": metrics["fields"]["overall"]["mean"],
                      "traces": summarise_traces(run_dir / "traces") if (run_dir / "traces").is_dir() else None},
                     indent=1))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
