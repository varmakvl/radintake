"""SDK vs batch CLI parity on the synthetic integration corpus (offline, real pipeline).

The same unseen-style folder goes through ``radintake.Pipeline.process_batch`` in-process and
through ``python -m radintake.cli batch`` as a subprocess; status, flags, input_type and every
report value must match (trace ids and timings excluded).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import radintake

from .fixtures import build_corpus

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "schemas" / "cli_output.v1.json").read_text(encoding="utf-8"))


def _key(doc: dict) -> dict:
    report = doc.get("report") or {}
    return {"status": doc["status"], "flags": sorted(doc["flags"]), "input_type": doc["input_type"],
            "n_pages": doc["n_pages"], "error": doc["error"],
            "report": {k: (v or {}).get("value") for k, v in report.items()}}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    inp = tmp_path_factory.mktemp("sdk_inputs")
    cli_out = tmp_path_factory.mktemp("sdk_cli_out")
    sdk_out = tmp_path_factory.mktemp("sdk_sdk_out")
    build_corpus(inp)
    env = {**os.environ, "RADINTAKE_OFFLINE": "1", "RADINTAKE_TRACE_DIR": str(cli_out / "_traces"),
           "RADINTAKE_DATA_DIR": str(cli_out / "_data")}
    proc = subprocess.run(
        [sys.executable, "-m", "radintake.cli", "batch", "--in", str(inp), "--out", str(cli_out)],
        capture_output=True, text=True, env=env, timeout=1800, cwd=ROOT, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    files = sorted(p for p in inp.iterdir() if p.is_file())
    pipe = radintake.Pipeline(radintake.RadintakeConfig(offline=True))
    results = pipe.process_batch(files)
    in_memory = radintake.Pipeline(radintake.RadintakeConfig(offline=True))
    from_bytes = [in_memory.process(p.read_bytes(), input_name=p.name) for p in files]
    written = radintake.Pipeline(radintake.RadintakeConfig(offline=True),
                                 output_dir=sdk_out).process_batch(files)
    return {"files": files, "cli_out": cli_out, "sdk_out": sdk_out, "results": results,
            "from_bytes": from_bytes, "written": written, "proc": proc}


def test_sdk_matches_cli_on_every_fixture(runs):
    assert len(runs["results"]) == len(runs["files"]) >= 7
    for f, r in zip(runs["files"], runs["results"], strict=True):
        cli_doc = json.loads((runs["cli_out"] / f"{f.name}.json").read_text(encoding="utf-8"))
        sdk_doc = r.model_dump(mode="json")
        Draft202012Validator(SCHEMA).validate(sdk_doc)
        assert sdk_doc["input"] == cli_doc["input"] == f.name
        assert _key(sdk_doc) == _key(cli_doc), f.name


def test_bytes_inputs_match_path_inputs(runs):
    for f, a, b in zip(runs["files"], runs["results"], runs["from_bytes"], strict=True):
        assert _key(a.model_dump(mode="json")) == _key(b.model_dump(mode="json")), f.name


def test_sdk_output_dir_writes_the_cli_documents(runs):
    for f, r in zip(runs["files"], runs["written"], strict=True):
        doc = json.loads((runs["sdk_out"] / f"{f.name}.json").read_text(encoding="utf-8"))
        assert doc == r.model_dump(mode="json")
        cli_doc = json.loads((runs["cli_out"] / f"{f.name}.json").read_text(encoding="utf-8"))
        assert _key(doc) == _key(cli_doc), f.name


def test_cli_summary_counts_match_sdk(runs):
    summary = json.loads(runs["proc"].stdout.strip().splitlines()[-1])
    statuses = [r.status for r in runs["results"]]
    assert summary["inputs"] == len(statuses)
    for k in ("ok", "needs_review", "error"):
        assert summary[k] == statuses.count(k)
