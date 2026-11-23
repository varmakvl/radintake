"""Unit tests for the planner state machine, stage registry, pipeline and batch CLI."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from radintake.cli import main
from radintake.config import load_config
from radintake.orchestrator import stages
from radintake.orchestrator.planner import Planner
from radintake.orchestrator.stages import Stage, StageContext
from radintake.pipeline import process_file

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "schemas" / "cli_output.v1.json").read_text())


@pytest.fixture
def registry_guard():
    """Remove any stage a test registers."""
    before = set(stages.registered())
    yield
    for name in set(stages.registered()) - before:
        stages.unregister(name)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("RADINTAKE_OFFLINE", "1")
    monkeypatch.setenv("RADINTAKE_TRACE_DIR", str(tmp_path / "_traces"))


def _pngs(folder: Path, n: int) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = folder / f"page_{i}.png"
        Image.new("L", (300, 400), 250).save(p)
        out.append(p)
    return out


def _steps(trace_dir: Path, case: str) -> list[dict]:
    (f,) = list(trace_dir.rglob(f"{case}.jsonl"))
    return [r["data"] for r in map(json.loads, f.read_text().splitlines()) if r["kind"] == "step"]


def _ctx(tmp_path, cfg, deadline_s=60.0) -> StageContext:
    return StageContext(path=tmp_path / "x.png", input_name="x.png", input_type="scan",
                        input_sha256="0" * 64, out_dir=tmp_path, config=cfg, trace=None,
                        deadline=time.monotonic() + deadline_s)


def test_stage_timeout_flags_and_batch_continues(tmp_path, registry_guard):
    calls = []

    @stages.register_stage("slow_test", applies_to={"scan"}, order=99, timeout_s=0.3, retries=0)
    def slow(ctx):
        calls.append(ctx.input_name)
        time.sleep(2.0)
        return {"report": {"never": "merged"}}

    inp, out = tmp_path / "in", tmp_path / "out"
    _pngs(inp, 2)
    t0 = time.monotonic()
    assert main(["batch", "--in", str(inp), "--out", str(out)]) == 0
    assert time.monotonic() - t0 < 3.5  # each slow stage abandoned after ~0.3 s
    assert sorted(calls) == ["page_0.png", "page_1.png"]  # second file still processed
    v = Draft202012Validator(SCHEMA)
    for i in range(2):
        doc = json.loads((out / f"page_{i}.png.json").read_text())
        assert not list(v.iter_errors(doc))
        assert "timeout" in doc["flags"]
        assert doc["status"] == "needs_review"  # load succeeded -> partial, not error
        assert doc["report"] is None  # the timed-out stage's result is discarded
        assert [s["status"] for s in _steps(tmp_path / "_traces", f"page_{i}.png")] == ["ok", "timeout"]


def test_failing_stage_is_retried_then_succeeds(tmp_path, registry_guard):
    attempts = {"n": 0}

    @stages.register_stage("flaky_test", applies_to={"scan"}, order=99, retries=2)
    def flaky(ctx):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient")
        return {"phi": {"masked_artifacts": [], "detections": []}}

    cfg = load_config()
    cfg.budgets = replace(cfg.budgets, backoff_s=0.01)
    (png,) = _pngs(tmp_path / "in", 1)
    doc = process_file(png, tmp_path / "out", config=cfg)
    assert attempts["n"] == 2
    assert doc["phi"] == {"masked_artifacts": [], "detections": []}
    assert "timeout" not in doc["flags"] and doc["error"] is None
    steps = [s for s in _steps(tmp_path / "_traces", "page_0.png") if s["stage"] == "flaky_test"]
    assert [(s["attempt"], s["status"]) for s in steps] == [(1, "failed"), (2, "ok")]


def test_noncritical_failure_flags_review_critical_failure_errors(tmp_path):
    cfg = load_config()
    cfg.budgets = replace(cfg.budgets, backoff_s=0.0, retries=1)

    def boom(ctx):
        raise ValueError("bad page")

    ok = Stage("ok", lambda ctx: {"n_pages": 1}, frozenset({"scan"}), order=1)
    soft = Stage("soft", boom, frozenset({"scan"}), order=2, failure_flag="preprocess_failed")
    hard = Stage("hard", boom, frozenset({"scan"}), order=3, critical=True)
    after = Stage("after", lambda ctx: {"report": {}}, frozenset({"scan"}), order=4)

    ctx = _ctx(tmp_path, cfg)
    res = Planner(cfg).run(ctx, [ok, soft])
    assert res.status == "needs_review" and "preprocess_failed" in ctx.flags
    assert res.stage_status == {"ok": "ok", "soft": "failed"}

    ctx = _ctx(tmp_path, cfg)
    res = Planner(cfg).run(ctx, [ok, hard, after])
    assert res.status == "error" and "bad page" in res.error
    assert res.stage_status["after"] == "skipped" and "report" not in ctx.data
    assert res.steps == 1 + 2  # ok once, hard twice (1 retry)


def test_step_budget_and_case_deadline(tmp_path):
    cfg = load_config()
    cfg.budgets = replace(cfg.budgets, max_steps=2)
    plan = [Stage(f"s{i}", lambda ctx: None, frozenset({"scan"}), order=i) for i in range(4)]
    ctx = _ctx(tmp_path, cfg)
    res = Planner(cfg).run(ctx, plan)
    assert res.steps == 2 and "timeout" in ctx.flags and res.status == "needs_review"

    cfg = load_config()
    ctx = _ctx(tmp_path, cfg, deadline_s=-1.0)
    res = Planner(cfg).run(ctx, plan)
    assert res.steps == 0 and res.status == "error" and "deadline" in res.error


def test_plan_is_per_input_type(registry_guard):
    @stages.register_stage("dicom_only_test", applies_to={"dicom"}, order=5)
    def d(ctx):
        return None

    assert "dicom_only_test" in [s.name for s in stages.plan_for("dicom")]
    assert "dicom_only_test" not in [s.name for s in stages.plan_for("pdf")]
    assert "load" in [s.name for s in stages.plan_for("pdf")]
    with pytest.raises(ValueError):
        stages.register_stage("bad", applies_to={"video"})


def test_junk_input_gives_error_json(tmp_path):
    p = tmp_path / "in" / "notes.txt"
    p.parent.mkdir()
    p.write_text("hello")
    doc = process_file(p, tmp_path / "out")
    assert doc["status"] == "error" and doc["error"] and doc["input_type"] == "unknown"
    assert json.loads((tmp_path / "out" / "notes.txt.json").read_text()) == doc


def test_cli_invalid_args_nonzero(tmp_path, capsys):
    assert main(["batch", "--in", str(tmp_path / "missing"), "--out", str(tmp_path / "o")]) != 0
    assert main(["nope"]) != 0
    assert main(["batch", "--in", str(tmp_path)]) != 0
    capsys.readouterr()
