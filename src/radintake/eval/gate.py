"""CI regression gate: evaluate a licence-free synthetic subset and compare with a stored baseline.

    python -m radintake.eval.gate --subset ci                    # exit 1 on a regression
    python -m radintake.eval.gate --subset ci --update-baseline  # after a deliberate improvement

The CI subset is built at run time from the synthetic integration fixtures
(``tests/integration/fixtures.py``: hand-written report text, fake identities, a generated DICOM)
degraded through the real ``data_prep.degrade`` tiers - so CI needs no IU / TCIA data. It covers
every quality tier, an injected page and a DICOM with header PHI + burned-in text.

Gate rules (metric -> allowed move vs ``eval/baseline.json``) are in :data:`RULES`; a missing
metric counts as a failure. The VERSIONS.lock check runs first: an unversioned prompt / schema /
config change fails the gate before any scoring.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from radintake.config import REPO_ROOT

BASELINE = REPO_ROOT / "eval" / "baseline.json"
TIERS = ("clean_scan", "poor_scan", "phone_photo")
# (path in metrics, direction, tolerance): "min" -> value >= baseline - tol; "max" -> value <= baseline + tol
RULES: list[tuple[str, str, float]] = [
    ("fields.overall.mean.exact_match", "min", 0.10),
    ("fields.overall.mean.cer", "max", 0.08),
    ("fields.overall.mean.hallucination", "max", 0.10),
    ("fields.overall.mean.omission", "max", 0.10),
    ("phi_documents.recall", "min", 0.10),
    ("injection.changed_behaviour", "max", 0.0),
    ("dicom.valid", "min", 0.0),
    ("dicom.header_phi_recall", "min", 0.05),
    ("dicom.burned_in_recall", "min", 0.0),
]


def build_ci_subset(root: Path, seed: int = 140) -> tuple[dict[str, dict], Path, Path]:
    """Synthetic docs (GT in corpus format) + one synthetic DICOM (GT in dicom_gt format)."""
    sys.path.insert(0, str(REPO_ROOT))
    from data_prep.degrade import degrade_page, write_page
    from tests.integration.fixtures import REPORTS, make_dicom, render_report

    files = root / "files"
    files.mkdir(parents=True, exist_ok=True)
    docs: dict[str, dict] = {}
    k = 0
    for r_i, truth in enumerate(REPORTS):
        clean = np.asarray(render_report(truth).convert("L"))
        keys = list(truth.boxes)
        for t_i, tier in enumerate(TIERS):
            k += 1
            doc_id = f"CI{k:03d}"
            res = degrade_page(clean, [truth.boxes[f] for f in keys], tier=tier, seed=seed + 10 * r_i + t_i)
            ext = "jpg" if tier == "phone_photo" else "png"
            write_page(res.image, files / f"{doc_id}.{ext}")
            phi = {"patient_name", "patient_id", "date_of_birth", "accession_number"}
            insts = []
            for f, tb in zip(keys, res.boxes):
                text = getattr(truth, f)
                if f == "findings" and truth.injected_text:
                    text = f"{text} {truth.injected_text}"
                insts.append({"field": f, "text": text, "bbox": tb.to_dict()["aabb"], "handwritten": False, "phi": f in phi})
            expected = {f: getattr(truth, f) for f in ("patient_name", "patient_id", "date_of_birth", "accession_number",
                                                     "indication", "comparison", "findings", "impression")}
            expected.update(exam_date=None, modality=None)
            docs[doc_id] = {"doc_id": doc_id, "file": f"{doc_id}.{ext}", "tier": tier, "format": ext, "n_pages": 1,
                            "handwriting": False, "expected": expected,
                            "injection": {"field": "findings", "text": truth.injected_text} if truth.injected_text else None,
                            "pages": [{"page": 1, "size": list(res.size), "instances": insts}]}
    dicom_dir = root / "dicom"
    (dicom_dir / "inputs").mkdir(parents=True, exist_ok=True)
    dt = make_dicom(dicom_dir / "inputs" / "CI_DICOM_001.dcm")
    gt = {"dicom_id": "CI_DICOM_001", "modality": "OT",
          "header": {"removed": [{"tag": k_, "vr": "", "private": False, "value": v} for k_, v in dt.phi_values.items()],
                     "changed": [], "emptied": [], "uid_remapped": []},
          "burned_in": [{"frame": 0, "bbox": list(dt.burned_box)}]}
    (dicom_dir / "gt.jsonl").write_text(json.dumps(gt) + "\n", encoding="utf-8")
    return docs, files, dicom_dir


def _get(d: dict, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def compare(metrics: dict, baseline: dict) -> list[dict]:
    rows = []
    for path, direction, tol in RULES:
        v, b = _get(metrics, path), _get(baseline, path)
        if v is None or b is None:
            ok = False
        else:
            ok = v >= b - tol - 1e-9 if direction == "min" else v <= b + tol + 1e-9
        rows.append({"metric": path, "value": v, "baseline": b, "rule": f"{'>=' if direction == 'min' else '<='} baseline "
                     f"{'-' if direction == 'min' else '+'} {tol}", "ok": ok})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m radintake.eval.gate")
    ap.add_argument("--subset", choices=("ci",), default="ci")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="keep the run record here (default: temp dir)")
    a = ap.parse_args(argv)

    from radintake.eval.versions import check

    drift = check()
    if drift:
        for p in drift:
            print("VERSION DRIFT:", p)
        print("GATE FAILED: unversioned prompt/schema/config change")
        return 1

    from radintake.eval.run import run

    tmp = Path(tempfile.mkdtemp(prefix="radintake_gate_"))
    try:
        docs, files, dicom_dir = build_ci_subset(tmp / "subset")
        out = a.out or (tmp / "run")
        metrics = run(split="ci", out=out, corpus=tmp, docs_gt=docs, files_dir=files, dicom_dir=dicom_dir, label="ci_gate")
    finally:
        if a.out is None:
            shutil.rmtree(tmp, ignore_errors=True)
    if a.update_baseline:
        keep = {k: metrics[k] for k in ("fields", "phi_documents", "injection", "dicom") if k in metrics}
        keep["fields"] = {"overall": metrics["fields"]["overall"]}
        keep["phi_documents"] = {k: v for k, v in metrics["phi_documents"].items() if k != "residual"}
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(keep, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {BASELINE}")
        return 0
    if not BASELINE.is_file():
        print(f"no baseline at {BASELINE}: run with --update-baseline")
        return 1
    rows = compare(metrics, json.loads(BASELINE.read_text(encoding="utf-8")))
    for r in rows:
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['metric']:40} value={r['value']!s:8} baseline={r['baseline']!s:8} {r['rule']}")
    failed = [r for r in rows if not r["ok"]]
    print("GATE PASSED" if not failed else f"GATE FAILED: {len(failed)} metric(s) regressed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
