"""API basics: health, problem+json errors, case flow, restart persistence, no PHI in logs."""

from __future__ import annotations

import io
import shutil
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from radintake.api.app import create_app
from radintake.config import load_config
from tests.integration.fixtures import REPORTS, render_report

needs_ocr = pytest.mark.skipif(not shutil.which("tesseract"), reason="tesseract not installed")


def _client(monkeypatch, data_dir):
    monkeypatch.setenv("RADINTAKE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("RADINTAKE_OFFLINE", "1")
    return TestClient(create_app(load_config()))


def _wait(c, cid, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        b = c.get(f"/cases/{cid}").json()
        if b["status"] in ("ok", "needs_review", "error"):
            return b
        time.sleep(0.3)
    pytest.fail("case did not finish")


def test_health_and_problem_json(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"
        r = c.get("/cases/nope")
        assert r.status_code == 404 and r.headers["content-type"].startswith("application/problem+json")
        assert r.json()["status"] == 404
        r = c.post("/cases")  # missing file
        assert r.status_code == 422 and r.headers["content-type"].startswith("application/problem+json")
        r = c.post("/query", json={"question": "x"})
        assert r.status_code in (200, 503)
        if r.status_code == 503:
            assert r.headers["content-type"].startswith("application/problem+json")


def test_bad_upload_and_zip(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as c:
        r = c.post("/cases", files={"file": ("x.txt", b"hello")})
        assert r.status_code == 202
        assert _wait(c, r.json()["case_id"])["status"] == "error"
        r = c.post("/cases", files={"file": ("../evil.zip", b"not a zip")})
        assert r.status_code == 400
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../../a.txt", "one")
            zf.writestr("b.txt", "two")
        r = c.post("/cases", files={"file": ("bundle.zip", buf.getvalue())})
        assert r.status_code == 202 and len(r.json()["case_ids"]) == 2
        for cid in r.json()["case_ids"]:
            assert _wait(c, cid)["status"] == "error"
        assert not (tmp_path.parent / "a.txt").exists()


def test_cases_and_review_queue_survive_restart(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as c:
        cid = c.post("/cases", files={"file": ("x.txt", b"hello")}).json()["case_id"]
        first = _wait(c, cid)
    with _client(monkeypatch, tmp_path) as c2:  # same RADINTAKE_DATA_DIR
        again = c2.get(f"/cases/{cid}").json()
        assert again["status"] == first["status"] and again["trace_id"] == first["trace_id"]
        assert c2.get("/review-queue").status_code == 200


def test_review_queue_lists_needs_review_cases(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as c:
        store = c.app.state.store
        cid = store.create("r.pdf")
        store.finish(cid, {"status": "needs_review", "flags": ["omission"], "trace_id": "t-1", "error": None})
        items = c.get("/review-queue").json()["items"]
        assert [i["case_id"] for i in items] == [cid] and items[0]["reasons"] == ["omission"]


@needs_ocr
def test_report_clean_png_reaches_final_status_without_phi_in_logs(monkeypatch, tmp_path, capfd):
    p = tmp_path / "report_clean.png"
    render_report(REPORTS[0]).save(p)
    with _client(monkeypatch, tmp_path / "data") as c:
        with p.open("rb") as fh:
            cid = c.post("/cases", files={"file": ("report_clean.png", fh)}).json()["case_id"]
        body = _wait(c, cid)
    assert body["status"] in ("ok", "needs_review")
    assert body["trace_id"] and "flags" in body
    logs = capfd.readouterr()
    for s in REPORTS[0].phi_strings:
        assert s not in logs.out + logs.err
